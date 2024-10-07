# lancio lo script dal terminale: 
# python data_analysis.py
# se da errore lanciare i seguenti comandi da terminale:
# pip install pandas
# pip install scipy


# pandas è la libreria per manipolare i csv
import pandas as pd
# scipy è la libreria per le statistiche
from scipy import stats

# definisco un ciclo for per scorrere tutti i file nella cartella
# range(2013,2024) genera una lista di numeri interi da 2013 a 2023 (l'ultimo è escluso)
for year in range(2013,2024):

    # stampo l'anno
    print(year)

    # leggo i dati relativi all'anno corrente e li importo in un dataframe
    # il dataframe contiene le colonne del CSV
    df = pd.read_csv(f"data/{year}_AL007.csv")

    # converto la colonna date in un formato data leggibile da pandas
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

    # Per raggruppare i dati setto l'indice del dataframe
    # inplace = True significa che le modifiche vengono fatte sul dataframe originale
    # inplace = False restituisce un nuovo dataframe e non modifica quello originale
    # un altro modo di scrivere il codice sotto è df = df.set_index('date', inplace = False)
    df.set_index('date', inplace=True)

    # Raggruppo i dati per mese
    # Visto che ho settato l'indice alla colonna 'date' posso usare df.index per indicare la colonna
    grouped = df.groupby(df.index.month)

    # Calcolo le metriche
    mean = grouped['T'].mean()
    # uso _ finale perché max è una parola chiave di Python
    max_ = grouped['T'].max()
    # uso _ finale perché min è una parola chiave di Python 
    min_ = grouped['T'].min()

    # Calcolo la moda applicando una funzione
    # lambda x permette di calcolare funzioni rapide
    # uso la funzione mode(x) fornita dalla libreria stats per calcolare la moda
    # la funzione mode(x) restituisce un oggetto di tipo mode che contiene una lista
    mode_ = grouped['T'].agg(lambda x: stats.mode(x, keepdims=False).mode)

    # Creo un nuovo dataframe con i risultati
    result = pd.DataFrame({
        'Year' : [year for _ in range(len(mean.index))],
        'Month': mean.index,
        # uso .values per accedere a tutti i valori di una colonna
        'Mean': mean.values,
        'Max': max_.values,
        'Min': min_.values,
        'Mode': mode_.values
    })

    # salvo il dataframe come CSV
    result.to_csv(f"output/{year}.csv", index=False)