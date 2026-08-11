"""
Generates a synthetic customer churn dataset, modeled after the well-known
Telco Customer Churn dataset structure. Correlations between features and
churn are injected on purpose (e.g. month-to-month contracts and low tenure
increase churn probability) so the trained model has real signal to learn.

Run:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 5000


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])

    tenure = rng.integers(0, 73, size=n_samples)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.24, 0.21],
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22]
    )
    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_samples),
    )

    def dependent_internet_feature():
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n_samples),
        )

    online_security = dependent_internet_feature()
    online_backup = dependent_internet_feature()
    device_protection = dependent_internet_feature()
    tech_support = dependent_internet_feature()
    streaming_tv = dependent_internet_feature()
    streaming_movies = dependent_internet_feature()

    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n_samples,
    )

    base_charge = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
    addon_cost = (
        (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    ) * rng.uniform(4, 7, size=n_samples)
    monthly_charges = np.round(base_charge + addon_cost + rng.normal(0, 3, size=n_samples), 2)
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0)

    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 20, size=n_samples), 2)
    total_charges = np.clip(total_charges, 0, None)

    # ---- Inject real signal into churn probability ----
    churn_score = np.zeros(n_samples)
    churn_score += np.where(contract == "Month-to-month", 0.45, 0.0)
    churn_score += np.where(contract == "One year", 0.10, 0.0)
    churn_score += np.where(internet_service == "Fiber optic", 0.20, 0.0)
    churn_score += np.where(payment_method == "Electronic check", 0.15, 0.0)
    churn_score += np.where(tech_support == "No", 0.12, 0.0)
    churn_score += np.where(online_security == "No", 0.10, 0.0)
    churn_score += np.where(paperless_billing == "Yes", 0.05, 0.0)
    churn_score += np.where(senior_citizen == 1, 0.08, 0.0)
    churn_score += np.clip((12 - tenure) / 12, 0, 1) * 0.35
    churn_score += np.clip((monthly_charges - 70) / 100, 0, 1) * 0.15
    churn_score -= np.where(partner == "Yes", 0.05, 0.0)
    churn_score -= np.where(dependents == "Yes", 0.05, 0.0)
    churn_score += rng.normal(0, 0.12, size=n_samples)  # noise

    churn_prob = 1 / (1 + np.exp(-6 * (churn_score - 0.78)))  # sigmoid squashing
    churn = rng.binomial(1, churn_prob)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i:05d}" for i in range(n_samples)],
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn_label,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    output_path = "data/telco_churn.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
