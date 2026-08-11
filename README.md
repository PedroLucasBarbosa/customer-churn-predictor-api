# Customer Churn Predictor API

API REST que prevê a probabilidade de um cliente cancelar o serviço (churn), a partir de dados de conta e uso, usando Machine Learning (scikit-learn) servido via FastAPI e containerizado com Docker.

Projeto de portfólio com foco em **Data + Software Development + Cloud**: cobre todo o ciclo de um produto de ML, da geração/preparação de dados até o deploy de uma API pronta para produção.

## 🧠 Sobre o modelo

- **Dataset:** sintético, gerado em `data/generate_data.py`, seguindo a estrutura e as correlações do conhecido dataset *Telco Customer Churn* (tenure, tipo de contrato, serviços contratados, forma de pagamento etc.)
- **Modelos comparados:** Logistic Regression vs Random Forest (ambos com pré-processamento em `Pipeline` do scikit-learn)
- **Modelo escolhido automaticamente:** o de maior ROC-AUC no conjunto de teste
- **Métricas atuais:** ver `models/model_metadata.json` após treinar (ou consultar `GET /model-info` com a API rodando)

## 📁 Estrutura do projeto

```
customer-churn-predictor-api/
├── data/
│   ├── generate_data.py     # gera o dataset sintético
│   └── telco_churn.csv      # dataset gerado (não versionar em produção real)
├── src/
│   ├── main.py               # aplicação FastAPI
│   ├── train.py               # treino e comparação de modelos
│   └── schemas.py             # schemas Pydantic (request/response)
├── models/
│   ├── churn_model.joblib     # pipeline treinado (pré-processamento + modelo)
│   └── model_metadata.json    # métricas e metadados do modelo
├── tests/
│   └── test_api.py            # testes automatizados da API
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Como rodar localmente

```bash
# 1. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Gerar o dataset (se ainda não existir)
python data/generate_data.py

# 4. Treinar o modelo
python src/train.py

# 5. Subir a API
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Depois, acesse a documentação interativa em **http://localhost:8000/docs**.

## 🐳 Como rodar com Docker

```bash
# Build da imagem
docker build -t churn-predictor-api .

# Rodar o container
docker run -p 8000:8000 churn-predictor-api
```

> O modelo já treinado (`models/churn_model.joblib`) é copiado para dentro da imagem — não é necessário treinar novamente no container.

## 📡 Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/` | Informações básicas da API |
| GET | `/health` | Health check (status, modelo carregado) |
| GET | `/model-info` | Tipo de modelo, versão e métricas |
| POST | `/predict` | Recebe dados do cliente e retorna previsão de churn |

### Exemplo de requisição — `POST /predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 427.5
  }'
```

### Exemplo de resposta

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.7834,
  "risk_level": "High"
}
```

## ✅ Testes

```bash
pytest tests/ -v
```

## ☁️ Próximos passos (deploy na AWS)

- [ ] Subir a imagem Docker para o Amazon ECR
- [ ] Fazer deploy em uma instância EC2 (ou ECS/Fargate)
- [ ] Configurar variáveis de ambiente e logs (CloudWatch)
- [ ] Adicionar autenticação (API Key) no endpoint `/predict`
- [ ] Monitorar model drift ao longo do tempo

## 🛠️ Tecnologias utilizadas

Python · scikit-learn · pandas · FastAPI · Pydantic · Docker · pytest
