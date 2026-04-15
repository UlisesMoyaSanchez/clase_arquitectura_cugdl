# Clasificación de Perros y Gatos con PyTorch Lightning

Proyecto de clasificación binaria (perros / gatos) usado como material didáctico para la clase de **Arquitecturas de Redes Neuronales** del grupo [CUGDL](https://github.com/cugdl).

Migrado a **PyTorch Lightning** con `LightningCLI`, split automático train/val/test, versionado de logs en TensorBoard, tracking de experimentos con **MLFlow** y pipelines reproducibles con **DVC**.

---

## Estructura del proyecto

```
.
├── configs/
│   ├── fit.yaml          # Hyperparámetros de entrenamiento
│   ├── fit_mlflow.yaml   # Entrenamiento con MLFlow logger
│   └── val.yaml          # Configuración de validación / test
├── data/
│   ├── dogs/             # Imágenes de perros (.jpg)
│   └── cats/             # Imágenes de gatos (.jpg)
├── models/
│   └── best_model.ckpt   # Mejor checkpoint (auto-guardado)
├── runs/                 # Logs de TensorBoard (versionados)
│   └── dogcat/
│       ├── version_0/
│       ├── version_1/
│       └── ...
├── config.py             # Constantes legadas (referencia)
├── dataset.py            # DogCatDataset + DogCatDataModule
├── main.py               # Punto de entrada (LightningCLI)
├── model.py              # DogCatModule (LightningModule)
├── utils.py              # Utilidades (set_seed, etc.)
├── create_data.sh        # Script para descargar el dataset
├── dvc.yaml              # Pipeline reproducible (DVC)
└── requirements.txt
```

---

## Requisitos

### Entorno Conda recomendado

```bash
conda activate pytorch   # PyTorch 2.7+ con CUDA 12.6
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

`requirements.txt` incluye:

| Paquete | Propósito |
|---|---|
| `torch` / `torchvision` | Framework de deep learning |
| `lightning` | Abstracción de entrenamiento (Trainer, CLI) |
| `torchmetrics` | Métricas (Accuracy, etc.) |
| `jsonargparse[signatures]` | Requerido por `LightningCLI` |
| `tensorboard` | Visualización de logs |
| `pillow` | Carga de imágenes |
| `numpy` | Operaciones numéricas |
| `mlflow` | Tracking de experimentos |
| `dvc` | Pipelines reproducibles y versionado de datos |

---

## Preparar el dataset

El script descarga 100 imágenes de perros y 100 de gatos automáticamente:

```bash
bash create_data.sh
```

Las imágenes quedan en `data/dogs/` y `data/cats/`. El dataset se divide en:

| Split | Proporción | ~Imágenes |
|---|---|---|
| Train | 70 % | 135 |
| Val | 15 % | 29 |
| Test | 15 % | 30 |

> La semilla es fija (`seed_everything: 42`) para reproducibilidad.

---

## Uso

### Entrenamiento

```bash
python main.py fit --config configs/fit.yaml
```

Cada ejecución crea automáticamente una nueva versión en `runs/dogcat/version_N/`.

#### Opciones útiles por línea de comandos

```bash
# Cambiar número de épocas
python main.py fit --config configs/fit.yaml --trainer.max_epochs 20

# Usar ResNet18 preentrenada
python main.py fit --config configs/fit.yaml --model.backbone resnet18

# Mixed precision (más rápido en GPU Ampere+)
python main.py fit --config configs/fit.yaml --trainer.precision 16-mixed

# Smoke test rápido (2 batches, 1 época)
python main.py fit --config configs/fit.yaml \
    --trainer.max_epochs 1 \
    --trainer.limit_train_batches 2 \
    --trainer.limit_val_batches 2
```

### Validación

```bash
python main.py validate --config configs/val.yaml
```

Usa el checkpoint definido en `configs/val.yaml` (`ckpt_path: models/best_model.ckpt`).

```bash
# Apuntar a un checkpoint específico
python main.py validate --config configs/val.yaml \
    --ckpt_path models/best_model-v1.ckpt
```

### Test

```bash
python main.py test --config configs/val.yaml
```

### Ver todos los argumentos disponibles

```bash
python main.py fit --help
python main.py fit --print_config          # imprime config actual como YAML
```

---

## TensorBoard

```bash
tensorboard --logdir runs
```

Abre [http://localhost:6006](http://localhost:6006) en el navegador.

Métricas registradas por época:

| Métrica | Descripción |
|---|---|
| `train/loss` | Pérdida en entrenamiento |
| `train/acc` | Accuracy en entrenamiento |
| `val/loss` | Pérdida en validación |
| `val/acc` | Accuracy en validación |
| `test/loss` | Pérdida en test |
| `test/acc` | Accuracy en test |

---

## Arquitecturas disponibles

Configurables desde `configs/fit.yaml` → `model.backbone`:

| Valor | Descripción |
|---|---|
| `simplecnn` | 3 bloques Conv+ReLU+Pool + 2 capas FC (2.2 M params con img\_size=64) |
| `resnet18` | ResNet-18 con pesos ImageNet, última capa reemplazada |

```yaml
# configs/fit.yaml
model:
  backbone: resnet18
  pretrained: true
```

---

## Configuración de entrenamiento (fit.yaml)

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `trainer.max_epochs` | `10` | Número de épocas |
| `trainer.accelerator` | `auto` | `cpu` / `gpu` / `mps` / `auto` |
| `trainer.precision` | `32-true` | `16-mixed` para AMP |
| `model.lr` | `0.001` | Learning rate (SGD) |
| `model.momentum` | `0.9` | Momentum (SGD) |
| `model.backbone` | `simplecnn` | Arquitectura del modelo |
| `data.batch_size` | `32` | Tamaño de lote |
| `data.img_size` | `64` | Tamaño de imagen (px) |
| `data.train_ratio` | `0.70` | Fracción para train |
| `data.val_ratio` | `0.15` | Fracción para val |
| `data.num_workers` | `4` | Workers del DataLoader |

Los callbacks activos son:

- **`ModelCheckpoint`** — guarda el mejor modelo según `val/acc`
- **`EarlyStopping`** — detiene si `val/loss` no mejora en 4 épocas
- **`LearningRateMonitor`** — registra el LR en TensorBoard

---

## Flujo general

```
create_data.sh
      │
      ▼
 data/dogs/  data/cats/
      │
      ▼
DogCatDataModule.setup()
  ├── train (70 %)  ← augmentation: flip + rotation
  ├── val   (15 %)  ← solo resize + normalize
  └── test  (15 %)  ← solo resize + normalize
      │
      ▼
DogCatModule (LightningModule)
  ├── training_step   → train/loss, train/acc
  ├── validation_step → val/loss,   val/acc
  └── test_step       → test/loss,  test/acc
      │
      ▼
Trainer (LightningCLI)
  ├── TensorBoardLogger → runs/dogcat/version_N/
  ├── ModelCheckpoint   → models/best_model.ckpt
  └── EarlyStopping
```

---

## MLFlow — Tracking de experimentos

[MLFlow](https://mlflow.org/) permite registrar métricas, hiperparámetros y artefactos (modelos) de
cada corrida experimental. El proyecto incluye una configuración lista para usar.

### Instalación

```bash
pip install mlflow
```

### Entrenar con MLFlow logger

```bash
python main.py fit --config configs/fit_mlflow.yaml
```

Este comando utiliza `MLFlowLogger` en lugar de `TensorBoardLogger`. Los resultados se
almacenan en el directorio local `mlruns/`.

### Ver la UI de MLFlow

```bash
mlflow ui --backend-store-uri mlruns
```

Abre [http://localhost:5000](http://localhost:5000) en el navegador para explorar:

- Métricas por época (`train/loss`, `train/acc`, `val/loss`, `val/acc`)
- Hiperparámetros del modelo y datos
- Artefactos (checkpoints del modelo)
- Comparación entre corridas

### Combinar MLFlow con overrides de CLI

```bash
# Cambiar backbone y LR, logueando todo en MLFlow
python main.py fit --config configs/fit_mlflow.yaml \
    --model.backbone resnet18 \
    --model.lr 0.0005

# Smoke test rápido con MLFlow
python main.py fit --config configs/fit_mlflow.yaml \
    --trainer.max_epochs 1 \
    --trainer.limit_train_batches 2 \
    --trainer.limit_val_batches 2
```

### Usar un servidor MLFlow remoto

Si tienes un servidor MLFlow centralizado, cambia `tracking_uri` en el YAML:

```yaml
# configs/fit_mlflow.yaml
trainer:
  logger:
    init_args:
      tracking_uri: http://mlflow-server:5000
```

---

## DVC — Pipelines reproducibles

[DVC](https://dvc.org/) (Data Version Control) permite definir pipelines reproducibles
y versionar datos y modelos de forma eficiente.

### Instalación

```bash
pip install dvc
```

### Inicializar DVC (solo la primera vez)

```bash
dvc init
```

Esto crea los archivos `.dvc/` y `.dvcignore`.

### Pipeline definido

El archivo `dvc.yaml` define dos etapas:

| Etapa | Comando | Dependencias | Salidas |
|---|---|---|---|
| `train` | `python main.py fit --config configs/fit.yaml` | `data/`, configs, código fuente | `models/best_model.ckpt` |
| `evaluate` | `python main.py test --config configs/val.yaml` | checkpoint, `data/`, código fuente | — |

### Ejecutar el pipeline completo

```bash
dvc repro
```

DVC detecta automáticamente qué etapas necesitan re-ejecutarse según los cambios
en dependencias (datos, código, configuración).

### Ejecutar una etapa específica

```bash
# Solo entrenar
dvc repro train

# Solo evaluar
dvc repro evaluate
```

### Ver el grafo del pipeline

```bash
dvc dag
```

```
+-------+
| train |
+-------+
     *
     *
     *
+----------+
| evaluate |
+----------+
```

### Versionar datos con DVC

```bash
# Trackear el directorio de datos
dvc add data/

# Esto crea data.dvc — commitear al repositorio
git add data.dvc .gitignore
git commit -m "Track dataset con DVC"

# Subir datos a un remote (S3, GCS, SSH, etc.)
dvc remote add -d myremote s3://mi-bucket/dvc-store
dvc push
```

### Ejemplo completo: flujo DVC + MLFlow

```bash
# 1. Inicializar (solo una vez)
dvc init

# 2. Versionar datos
dvc add data/
git add data.dvc .gitignore

# 3. Entrenar con MLFlow tracking
python main.py fit --config configs/fit_mlflow.yaml

# 4. Reproducir pipeline completo
dvc repro

# 5. Ver métricas en MLFlow
mlflow ui --backend-store-uri mlruns

# 6. Comparar experimentos
mlflow runs list --experiment-name dogcat
```

---

## Flujo general (con DVC + MLFlow)

```
create_data.sh
      │
      ▼
 data/dogs/  data/cats/
      │
  dvc add data/         ← versiona datos
      │
      ▼
dvc repro               ← ejecuta pipeline
      │
      ├── train stage
      │     └── python main.py fit --config configs/fit_mlflow.yaml
      │           ├── MLFlowLogger  → mlruns/   (métricas + artefactos)
      │           └── ModelCheckpoint → models/best_model.ckpt
      │
      └── evaluate stage
            └── python main.py test --config configs/val.yaml
      │
      ▼
mlflow ui               ← visualiza resultados en localhost:5000
```
