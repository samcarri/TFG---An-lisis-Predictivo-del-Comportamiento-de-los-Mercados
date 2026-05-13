"""
normalize_selftext.py
Normaliza el campo selftext del CSV nvda_processed.csv:
  - Decodifica HTML entities (&amp; &gt; &lt; etc.)
  - Elimina saltos de línea y espacios múltiples
  - Elimina markdown de Reddit (**, \\*, #, etc.)
  - Elimina URLs
  - Convierte a minúsculas
  - Rellena nulos con cadena vacía
"""

import re
import html
import pandas as pd

INPUT_PATH = "data/reddit_posts.csv"
OUTPUT_PATH = "data/reddit_posts.csv"  # sobreescribe; cambia si prefieres otro archivo


def normalize_selftext(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # 1. Decodificar HTML entities (&amp; -> &, &gt; -> >, etc.)
    text = html.unescape(text)

    # 2. Eliminar URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Eliminar markdown de Reddit: negrita/cursiva (**text**, *text*, ~~text~~)
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)

    # 4. Eliminar encabezados markdown (# Título)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # 5. Eliminar caracteres de escape de Reddit (\n, \t, \\)
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\", "")

    # 6. Reemplazar saltos de línea reales por espacio
    text = text.replace("\n", " ").replace("\r", " ")

    # 7. Eliminar caracteres especiales no alfanuméricos (conserva puntuación básica)
    text = re.sub(r"[^\w\s.,!?;:()\'\"-]", " ", text)

    # 8. Colapsar espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()

    # 9. Minúsculas
    text = text.lower()

    return text


def main():
    df = pd.read_csv(INPUT_PATH)

    print(f"Filas totales: {len(df)}")
    print(f"Nulos en selftext antes: {df['selftext'].isna().sum()}")

    df["selftext"] = df["selftext"].apply(normalize_selftext)

    print(f"Vacíos en selftext después: {(df['selftext'] == '').sum()}")
    print("\nEjemplo normalizado:")
    sample = df.loc[df["selftext"] != "", "selftext"].iloc[0]
    print(sample[:300])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nArchivo guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
