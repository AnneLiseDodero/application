"""
Predict the price per square metre of housing in Paris and its inner suburbs.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import os
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from skore import EstimatorReport

# =============================================================================
# CONFIGURATION
# =============================================================================

os.chdir("/home/onyxia/work/application")

load_dotenv()
jeton_api = os.environ["JETON_API"]

parser = argparse.ArgumentParser(description="Prédiction du prix au m² (DVF)")
parser.add_argument(
    "--n_trees", type=int, default=1500, help="nombre d'arbres du modèle"
)
args = parser.parse_args()

n_trees = args.n_trees

departements_paris = ["75", "92", "93", "94"]

SOURCES = {
    "dvf.parquet":
    "https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet",
    "filosofi_carreaux_200m_2021.parquet":
    "https://www.data.gouv.fr/api/1/datasets/r/55432374-a91d-43d0-923d-4514dc3eb951",
}

data_location = Path("data")
data_location.mkdir(exist_ok=True)


# =============================================================================
# FUNCTIONS
# =============================================================================


def download_sources(adresse: str, cible: Path) -> None:
    """Download the file at `adresse` and write it to `cible`."""
    r = requests.get(adresse, timeout=120)
    r.raise_for_status()
    with open(cible, "wb") as f:
        f.write(r.content)


def as_category(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to 'category' so that LightGBM treats them as categorical."""
    return df.astype("category")


def check_code_commune_formatting(df: pd.DataFrame) -> int:
    """Return the number of rows whose `code_commune` is not 5 characters long
    starting with `code_departement`."""
    con = duckdb.connect(database=":memory:")
    # DuckDB cannot read the geometry column
    con.register(
        "attributs", pd.DataFrame(df[["code_commune", "code_departement"]])
    )
    return con.sql("""
        SELECT COUNT(*)
        FROM attributs
        WHERE length(code_commune) <> 5
        OR NOT starts_with(code_commune, code_departement)
    """).fetchone()[0]


def check_missing_values(df: pd.DataFrame, column: str) -> int:
    """Return the number of missing values in `column`."""
    return int(df[column].isna().sum())


def check_unseen_categories(
    df_train: pd.DataFrame, df_test: pd.DataFrame, column: str
) -> set:
    """Return the categories of `column` present in test but not in train."""
    return set(df_test[column].dropna().unique()) - set(
        df_train[column].dropna().unique()
    )


def get_melodi_indicator(
    code: str,
    value_name: str,
    token: str,
    dataset: str = "DS_FILOSOFI_CC",
    dimension: str = "FILOSOFI_MEASURE",
    time_period: int = 2023,
    geo_level: tuple[str, ...] = ("COM", "ARM"),
) -> pd.DataFrame:
    """Fetch one indicator from the INSEE Melodi API, by municipality.

    Returns a DataFrame with columns `CODGEO` and `value_name`.
    """
    response = requests.get(
        f"https://api.insee.fr/melodi/data/{dataset}",
        params={
            "GEO": list(geo_level),
            dimension: code,
            "TIME_PERIOD": time_period,
            "maxResult": 40000,
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    response.raise_for_status()

    return pd.DataFrame(
        {
            "CODGEO": obs["dimensions"]["GEO"].split("-")[-1],
            value_name: obs["measures"]["OBS_VALUE_NIVEAU"].get("value"),
        }
        for obs in response.json()["observations"]
    )


# =============================================================================
# 1. DATA DOWNLOAD AND LOADING
# =============================================================================

print("Lecture des données ------------------------")

for nom, url in SOURCES.items():
    target = data_location / nom
    if target.exists():
        print(f"{nom:42s} déjà présent ({target.stat().st_size / 1e6:.0f} Mo)")
    else:
        print(f"Téléchargement de {nom} ...")
        download_sources(url, target)
        print(f"  -> {target.stat().st_size / 1e6:.0f} Mo")

dvf = gpd.read_parquet(data_location / "dvf.parquet")


# =============================================================================
# 2. DATA QUALITY CHECKS AND FILTERING
# =============================================================================

n_bad = check_code_commune_formatting(dvf)
if n_bad == 0:
    print("Test 'code_commune' OK: toujours 5 caractères commençant par le département")
else:
    print(f"Problème dans la colonne code_commune: {n_bad} codes mal formés.")

for col in dvf.columns:
    n_missing = check_missing_values(dvf, col)
    if n_missing > 0:
        print(f"{n_missing} valeurs manquantes pour la variable {col}")

dvf = dvf.loc[dvf["valeur_fonciere"] < 1e6]
dvf["annee"] = pd.to_datetime(dvf["date_mutation"]).dt.year

# Filter before the spatial join to avoid joining the whole of France
print("Restriction aux départements de la petite couronne")

dvf = dvf.loc[dvf["code_departement"].isin(departements_paris)]


# =============================================================================
# 3. ADDITIONAL DATA
# =============================================================================

# 3.1 Filosofi 200 m grid -----------------------------------------------------

filo_cols = [
    "idcar_200m",
    "ind",
    "men",
    "men_pauv",
    "men_prop",
    "ind_snv",
    "log_soc",
    "log_av45",
    "men_surf",
    "geometry",
]
filo = gpd.read_parquet(
    data_location / "filosofi_carreaux_200m_2021.parquet", columns=filo_cols
)
filo = filo.rename(
    columns={
        "men": "nbre_menages",
        "men_pauv": "nbre_menages_pauvres",
        "men_prop": "nbre_menages_prop",
        "ind_snv": "somme_nv_vie",
        "log_soc": "nbre_logements_sociaux",
        "log_av45": "nbre_logements_avant_1945",
        "men_surf": "surface_totale_logements",
    }
)

print(
    f"Reprojection de {dvf.shape[0]} de transactions immobilières du CRS "
    f"{dvf.crs.to_epsg()} à {filo.crs.to_epsg()}"
)

dvf = dvf.to_crs(filo.crs)
dvf = dvf.sjoin(filo, how="left", predicate="within")


# 3.2 Municipal Filosofi data (INSEE Melodi API) ------------------------------

# DVF codes Paris by arrondissement (ARM), not as a single municipality (COM):
# get_melodi_indicator requests both levels by default.
niveau_vie = get_melodi_indicator("MED_SL", "NIVVIE_MEDIAN", jeton_api)
taux_pauvrete = get_melodi_indicator("PR_MD60", "TAUX_PAUVRETE", jeton_api)


# 3.3 Merge with DVF ----------------------------------------------------------

for indicateur in (niveau_vie, taux_pauvrete):
    dvf = dvf.merge(
        indicateur, left_on="code_commune", right_on="CODGEO", how="left"
    ).drop(columns="CODGEO")

print(dvf.head(5))


# =============================================================================
# 4. DESCRIPTIVE STATISTICS
# =============================================================================

print("Statistiques descriptives ---------------------")

sns.histplot(data=dvf, x="valeur_fonciere")
sns.histplot(data=dvf, x="valeur_fonciere", log_scale=True)

grid = sns.scatterplot(data=dvf, x="surface_reelle_bati", y="valeur_fonciere")
grid.set(xscale="log", yscale="log")


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================

print("Feature engineering")

dvf["tx_pauvrete"] = dvf["nbre_menages_pauvres"] / dvf["nbre_menages"]
dvf["nv_vie_moyen"] = dvf["somme_nv_vie"] / dvf["nbre_menages"]
# In Filosofi, one household = one dwelling
dvf["part_proprietaires"] = dvf["nbre_menages_prop"] / dvf["nbre_menages"]
dvf["part_logements_sociaux"] = dvf["nbre_logements_sociaux"] / dvf["nbre_menages"]
dvf["part_logements_avant_1945"] = (
    dvf["nbre_logements_avant_1945"] / dvf["nbre_menages"]
)
dvf["surface_moyenne_logements"] = dvf["surface_totale_logements"] / dvf["nbre_menages"]
dvf["prix_m2"] = dvf["valeur_fonciere"] / dvf["surface_reelle_bati"]

# Extreme prices are mostly multi-lot sales or input errors (test R² 0.39 -> 0.61)
borne_basse, borne_haute = dvf["prix_m2"].quantile([0.01, 0.99])
dvf = dvf.loc[dvf["prix_m2"].between(borne_basse, borne_haute)]

# Global bounds are driven by Paris: also trim within each municipality
prix_m2_par_commune = dvf.groupby("code_commune")["prix_m2"]
dvf = dvf.loc[
    dvf["prix_m2"].between(
        prix_m2_par_commune.transform("quantile", 0.01),
        prix_m2_par_commune.transform("quantile", 0.99),
    )
]

dvf["log_surface"] = np.log(dvf["surface_reelle_bati"])

# DVF prices include land and outbuildings; Carrez/built-up ratio flags
# inconsistent surfaces (NaN outside co-ownership)
dvf["log_surface_terrain"] = np.log1p(dvf["surface_terrain"].fillna(0))
dvf["ratio_carrez"] = dvf["lot1_surface_carrez"] / dvf["surface_reelle_bati"]

# Distance to the Eiffel Tower as a centrality proxy (Lambert-93, in metres)
tour_eiffel = gpd.GeoSeries.from_xy([2.2945], [48.8584], crs="EPSG:4326").to_crs(
    "EPSG:2154"
)
distance_eiffel_km = (
    dvf.geometry.to_crs("EPSG:2154").distance(tour_eiffel.iloc[0]) / 1000
)
dvf["log_distance_eiffel"] = np.log1p(distance_eiffel_km)

prix_m2_commune = (
    dvf.groupby(["code_commune", "nom_commune"])["prix_m2"].mean().sort_values()
)

print("Bottom 5 communes (prix au m2 le plus bas) :")
print(prix_m2_commune.head(5))
print("Top 5 communes (prix au m2 le plus élevé) :")
print(prix_m2_commune.tail(5))


# =============================================================================
# 6. PIPELINE
# =============================================================================

numeric_features = [
    "log_surface",
    "log_distance_eiffel",
    "nombre_pieces_principales",
    "nombre_lots",
    "log_surface_terrain",
    "ratio_carrez",
    "nbre_menages",
    "tx_pauvrete",
    "nv_vie_moyen",
    "part_proprietaires",
    "part_logements_sociaux",
    "part_logements_avant_1945",
    "surface_moyenne_logements",
    "NIVVIE_MEDIAN",
    "TAUX_PAUVRETE",
]
categorical_features = ["code_commune", "type_local"]
features = list(set(numeric_features + categorical_features))

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ordinal",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
        ("as_category", FunctionTransformer(as_category)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        ("Preprocessing categorical", categorical_transformer, categorical_features),
    ]
).set_output(transform="pandas")

_pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            LGBMRegressor(
                n_estimators=n_trees,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=0,
                verbose=-1,
            ),
        ),
    ]
)

# Skewed target: fit on log(1 + price) so the most expensive sales do not dominate
pipe = TransformedTargetRegressor(regressor=_pipe, func=np.log1p, inverse_func=np.expm1)


# =============================================================================
# 7. TRAIN / TEST SPLIT
# =============================================================================

# Missing features are imputed in the pipeline: only drop rows without a target
dvf_start_data = dvf.dropna(subset=["prix_m2"])

X = dvf_start_data[features]
y = dvf_start_data["prix_m2"]
# weights = dvf_start_data["code_commune"] + "_" + dvf_start_data["annee"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
    X, y, test_size=0.1, random_state=0
)  # stratify = weights)

pd.concat([X_train, y_train], axis=1).to_csv("train.csv")
pd.concat([X_test, y_test], axis=1).to_csv("test.csv")

for col in categorical_features:
    unseen = check_unseen_categories(X_train, X_test, col)
    if unseen:
        print(f"Modalités de {col} absentes de l'apprentissage : {unseen}")
    else:
        print(f"Toutes les modalités de {col} du test sont connues du modèle")


# =============================================================================
# 8. TRAINING
# =============================================================================

pipe.fit(X_train, y_train)


# =============================================================================
# 9. EVALUATION
# =============================================================================

# 9.1 RMSE (overall and by municipality) --------------------------------------

rmse_train = root_mean_squared_error(y_train, pipe.predict(X_train))
rmse_test = root_mean_squared_error(y_test, pipe.predict(X_test))
print(f"RMSE sur l'apprentissage : {rmse_train:,.0f} €/m²")
print(f"RMSE sur le test : {rmse_test:,.0f} €/m²")

residuals = X_test[["code_commune"]].copy()
residuals["y_test"] = y_test
residuals["pred"] = pipe.predict(X_test)

rmse_commune = (
    residuals.groupby("code_commune")
    .apply(
        lambda d: root_mean_squared_error(d["y_test"], d["pred"]), include_groups=False
    )
    .rename("rmse")
    .reset_index()
)

cog = dvf_start_data[["code_commune", "nom_commune"]].drop_duplicates()

rmse_commune = rmse_commune.merge(cog, on="code_commune").sort_values("rmse")

print("Bottom 5 communes (RMSE la plus faible) :")
print(rmse_commune.head(5))
print("Top 5 communes (RMSE la plus élevée) :")
print(rmse_commune.tail(5))


# 9.2 skore report ------------------------------------------------------------

print("Évaluation skore ---------------------")

# skore refits a copy of the pipeline on the same split
report = EstimatorReport(
    clone(pipe),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

print(report)

metrics = report.metrics.summarize(data_source="both").frame()
print(metrics)

report.metrics.prediction_error().plot(kind="actual_vs_predicted")
plt.savefig("prediction_error.png", dpi=150, bbox_inches="tight")
plt.close()

importance = report.inspection.permutation_importance(data_source="test").frame()
print(importance)

print("Fin du script !")
