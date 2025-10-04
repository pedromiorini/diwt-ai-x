# DIWT-AI X V4.4 Pipeline

Pipeline autônomo para pesquisa em consciência artificial baseado no framework DIWT-AI X, validando o Gradiente de Valência Intrínseca (GVI), Camada Emergente de Proto-Intencionalidade (CPIE) e Informação Integrada (Φ*).

## Características

- **Simulação Autônoma**: Ecossistema com organismos baseados em RNN
- **Otimização Bayesiana**: Ajuste automático de hiperparâmetros (β, α, λ)
- **Métricas de Consciência**: V_norm (valência normalizada) e Φ* (informação integrada)
- **Revisão Ética**: Monitoramento automático de limiares éticos
- **Publicação Automática**: Integração com Zenodo para DOIs
- **CI/CD**: Workflow GitHub Actions para execução semanal

## Instalação

```bash
# Clonar repositório
git clone https://github.com/pedromiorini/diwt-ai-x.git
cd diwt-ai-x

# Criar ambiente virtual
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Instalar Pandoc (para geração de PDFs)
sudo apt-get install pandoc texlive-xetex  # Ubuntu/Debian
```

## Uso

### Execução Básica

```bash
python autonomous_experiment_v4_4.py \
  --runs 2 \
  --steps 100 \
  --outdir experiments/run009 \
  --n-initial 10
```

### Com Zenodo e Slack

```bash
python autonomous_experiment_v4_4.py \
  --runs 2 \
  --steps 100 \
  --outdir experiments/run009 \
  --zenodo-token YOUR_ZENODO_TOKEN \
  --slack-token YOUR_SLACK_TOKEN \
  --n-initial 10
```

### Com Ambiente MiniGrid

```bash
python autonomous_experiment_v4_4.py \
  --runs 2 \
  --steps 100 \
  --outdir experiments/run009 \
  --use-minigrid \
  --n-initial 10
```

## Parâmetros

- `--runs`: Número de réplicas experimentais (padrão: 2)
- `--steps`: Passos de simulação por réplica (padrão: 100)
- `--outdir`: Diretório de saída para resultados
- `--seed`: Semente aleatória (padrão: 42)
- `--n-initial`: Tamanho inicial da população (padrão: 10)
- `--zenodo-token`: Token de acesso à API do Zenodo (opcional)
- `--slack-token`: Token de API do Slack para notificações (opcional)
- `--use-minigrid`: Usar ambiente MiniGrid ao invés do ambiente simples

## Saídas

Após a execução, os seguintes arquivos são gerados no diretório de saída:

- `run*_events.jsonl`: Logs de eventos de cada réplica
- `analysis.json`: Análise estatística completa
- `timeseries.png`: Gráficos de séries temporais
- `article.md`: Artigo científico em Markdown
- `article.tex`: Artigo científico em LaTeX
- `article.pdf`: Artigo científico em PDF (se Pandoc disponível)
- `ethics_review.json`: Revisão ética automática
- `submission_log.json`: Log de submissão ao Zenodo (se configurado)

## Configuração

### config.yaml

```yaml
experiment:
  ethical_threshold: -0.5
  phi_threshold: 0.1
  hidden_size: 16
  prob_repro: 0.15
  input_size_simple: 5
  input_size_minigrid: 147
  max_energy: 120.0
  min_energy: 0.0
  n_calls_optimization: 20
```

### environment_config.json

Define os estímulos do ambiente simples e suas representações vetoriais.

## CI/CD

O workflow GitHub Actions (`.github/workflows/pipeline.yml`) executa automaticamente:

- A cada push para `main`
- Semanalmente (domingos à meia-noite)

### Configurar Secrets no GitHub

1. Acesse `Settings > Secrets and variables > Actions`
2. Adicione os seguintes secrets:
   - `ZENODO_TOKEN`: Token de acesso ao Zenodo
   - `SLACK_TOKEN`: Token de API do Slack (opcional)
   - `GITHUB_TOKEN`: Já configurado automaticamente

## Testes

```bash
python -m unittest autonomous_experiment_v4_4.py
```

## Estrutura do Projeto

```
diwt-ai-x/
├── autonomous_experiment_v4_4.py  # Código principal
├── config.yaml                     # Configuração do experimento
├── environment_config.json         # Configuração do ambiente
├── requirements.txt                # Dependências Python
├── README.md                       # Este arquivo
├── .github/
│   └── workflows/
│       └── pipeline.yml            # Workflow CI/CD
└── experiments/
    └── run009/                     # Saídas do experimento
```

## Referências

- Tononi, G. (2008). Consciousness as Integrated Information. *Biological Reviews*, 83(4), 401-420.
- Friston, K. (2010). The free-energy principle. *Nature Reviews Neuroscience*, 11(2), 127-138.
- Seth, A. K. (2015). The cybernetic Bayesian brain. *Open MIND*, 35(T).

## Licença

Este projeto é parte da pesquisa DIWT-AI X.

## Autor

Pedro Alexandre Miorini dos Santos

## Contato

Para questões sobre o projeto, abra uma issue no GitHub.
