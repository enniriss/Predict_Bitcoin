import pandas as pd
from datetime import datetime

def nettoyer_dataset(df):
    print("Début du nettoyage du dataset.")

    # Suppression des colonnes inutiles ou à forte cardinalité
    colonnes_a_supprimer = [
        "user_id", "country", "occupation", "ethnicity", "language_preference",
        "device_type", "preferred_payment_method", "product_category_preference",
        "shopping_time_of_day", "budgeting_style",
        "stress_from_financial_decisions", "overall_stress_level",
        "impulse_buying_score", "impulse_purchases_per_mounth",
        "cart_abandonment_rate", "daily_session_time_minutes", "app_usage_frequency"
    ]
    to_drop = [col for col in colonnes_a_supprimer if col in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"Colonnes supprimées : {to_drop}")

    # Encodage de gender
    if 'gender' in df.columns:
        df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1})
        df = df.drop(columns=['gender'])
        print("Colonne 'gender' encodée et supprimée.")

    # Encodage one-hot de urban_rural
    if 'urban_rural' in df.columns:
        df = pd.get_dummies(df, columns=['urban_rural'], drop_first=True)
        print("Colonnes 'urban_rural' encodées (one-hot) et supprimées.")

    # Encodage one-hot des colonnes catégorielles
    cat_cols = ['employment_status', 'education_level', 'relationship_status']
    existing_cat_cols = [col for col in cat_cols if col in df.columns]
    if existing_cat_cols:
        df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)
        print(f"Colonnes catégorielles encodées : {existing_cat_cols}")

    # Conversion des booléens en int
    bool_cols = ['has_children', 'loyalty_program_member', 'weekend_shopper', 'premium_subscription']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
            print(f"Colonne '{col}' convertie en int.")

    # Feature engineering sur la date
    if 'last_purchase_date' in df.columns:
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
        df['days_since_last_purchase'] = (pd.Timestamp.today() - df['last_purchase_date']).dt.days
        df['last_purchase_month'] = df['last_purchase_date'].dt.month
        df['last_purchase_year'] = df['last_purchase_date'].dt.year
        df['last_purchase_weekday'] = df['last_purchase_date'].dt.weekday
        df = df.drop(columns=['last_purchase_date'])
        print("Feature engineering sur 'last_purchase_date' effectué.")

    # Suppression des lignes avec valeurs nulles
    avant_na = len(df)
    df = df.dropna()
    print(f"Lignes supprimées pour valeurs nulles : {avant_na - len(df)}")

    # Suppression des doublons
    avant_dup = len(df)
    df = df.drop_duplicates()
    print(f"Lignes dupliquées supprimées : {avant_dup - len(df)}")

    # Suppression des outliers (méthode des 3 écarts-types)
    colonnes_numeriques = df.select_dtypes(include=['float64', 'int64']).columns
    avant_outliers = len(df)
    for col in colonnes_numeriques:
        if col != 'return_rate':  # Ne pas supprimer d'outliers sur la cible
            moyenne = df[col].mean()
            ecart_type = df[col].std()
            if ecart_type > 0:  # Éviter division par zéro
                df = df[(df[col] >= moyenne - 3 * ecart_type) & (df[col] <= moyenne + 3 * ecart_type)]
    print(f"Lignes supprimées pour outliers (méthode 3 écarts-types) : {avant_outliers - len(df)}")

    # Vérification finale : toutes les colonnes sont numériques
    non_numeric = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
    if non_numeric:
        print(f"ATTENTION : Colonnes non numériques restantes : {non_numeric}")
    else:
        print("Toutes les colonnes sont numériques.")

    print("Nettoyage terminé.")
    return df

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\Utilisateur\Documents\Cour\Data Science\GROUPE-EXAM-FINAL\Predict_Bitcoin\Data\e_commerce_shopper_behaviour_and_lifestyle.csv")
    df = nettoyer_dataset(df)
    print(df['return_rate'].describe())
    print(df['return_rate'].value_counts())
    
    # Analyse de la corrélation avec la cible
    corr_with_target = df.corr()['return_rate'].sort_values(ascending=False)
    print(corr_with_target)
    
    df.to_csv(r"C:\Users\Utilisateur\Documents\Cour\Data Science\GROUPE-EXAM-FINAL\Predict_Bitcoin\Data\e_commerce_shopper_behaviour_and_lifestyle_clean.csv", index=False)
    print("Dataset nettoyé sauvegardé dans 'e_commerce_shopper_behaviour_and_lifestyle_clean.csv'.")