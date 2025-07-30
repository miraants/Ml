from fastapi import FastAPI, Request
from pydantic import BaseModel
import pyodbc
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import uvicorn

# Démarrer l'app FastAPI
app = FastAPI()

# Modèle de données attendu en entrée
class ProfilUtilisateur(BaseModel):
    fonction: str
    ville_actuelle: str
    ville_voulue: str
    competences: str

# Connexion SQL
conn_str = (
    "Driver={SQL Server};"
    "Server=MDG-LT075;"
    "Database=MoovAstek_V1;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)
conn = pyodbc.connect(conn_str)

# Charger le modèle NLP une fois
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/recommander")
def recommander(profil: ProfilUtilisateur):
    # Charger les opportunités
    query = """
    SELECT o.ID, o.Titre, o.Summary, el.Contries AS city
    FROM Opportunity o
    LEFT JOIN ElementList el ON o.ID_Cities = el.ID_Element
    WHERE o.Status = 1;
    """
    df = pd.read_sql(query, conn)
    df.rename(columns={"Titre": "titre", "Summary": "summary"}, inplace=True)
    df.dropna(subset=["titre", "city", "summary"], inplace=True)

    # Fusion texte
    df["texte_complet"] = (
        df["titre"] + " - " +
        df["city"] + " - " +
        df["summary"]
    )

    # Texte utilisateur
    texte_utilisateur = (
        profil.fonction + " - " +
        profil.ville_voulue + " - " +
        profil.competences
    )

    # Similarité
    emb_user = model.encode(texte_utilisateur, convert_to_tensor=True)
    emb_offres = model.encode(df["texte_complet"].tolist(), convert_to_tensor=True)
    scores = util.cos_sim(emb_user, emb_offres)[0]
    df["score"] = scores.cpu().numpy()

    top = df.sort_values(by="score", ascending=False).head(1).iloc[0]

    return {
        "titre": top["titre"],
        "ville": top["city"],
        "resume": top["summary"],
        "score": float(top["score"])
    }


# Lancer avec uvicorn si exécuté directement
if __name__ == "__main__":
    uvicorn.run("ml_api:app", host="127.0.0.1", port=8000, reload=True)
