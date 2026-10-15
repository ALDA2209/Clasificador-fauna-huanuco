# 🐾 Clasificador de Fauna en Peligro — Huánuco

Proyecto académico de clasificación de imágenes mediante Transfer Learning (MobileNetV2), orientado a la identificación de mamíferos en peligro de extinción nativos de la región Huánuco, Perú. Desarrollado en el marco de la Línea 3 (Conservación de la biodiversidad y protección de fauna silvestre) del concurso **INNOBEST Huánuco 2026**.

## 🧬 Especies clasificadas

| Especie | Nombre científico |
|---|---|
| Capuchino de frente blanca | *Cebus yuracus* |
| Mono choro de cola amarilla | *Lagothrix flavicauda* |
| Oso de anteojos / Oso andino | *Tremarctos ornatus* |

## ⚙️ Tecnología

- **Modelo:** MobileNetV2 + Transfer Learning (Keras / TensorFlow)
- **Entrenamiento:** Google Colab con GPU T4, 100 épocas
- **Dataset:** imágenes obtenidas vía API pública de iNaturalist
- **Mejor val_accuracy:** 91.3%
- **Interfaz:** Streamlit

## 🚀 Cómo correrlo localmente

```bash
git clone https://github.com/ALDA2209/Clasificador-fauna-huanuco.git
cd Clasificador-fauna-huanuco
pip install -r requirements.txt --break-system-packages
python3 -m streamlit run app.py
```

Luego abre `http://localhost:8501` en tu navegador.

## 👤 Autor

Aldahir Pablo Silva Villar · SENATI Huánuco
