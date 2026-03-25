#!/bin/bash
# Script para descargar 100 imágenes de perros y 100 de gatos

# Verificar dependencias
if ! command -v jq &> /dev/null; then
    echo "ERROR: 'jq' no está instalado. Instálalo con: sudo apt install jq / brew install jq"
    exit 1
fi

mkdir -p data/dogs
mkdir -p data/cats

# ─── PERROS (Dog CEO API) ───────────────────────────────────────────
echo "Descargando 100 imágenes de perros..."

downloaded=0
attempts=0

while [ $downloaded -lt 100 ] && [ $attempts -lt 200 ]; do
    ((attempts++))

    # jq parsea el JSON de forma segura
    img_url=$(curl -sf "https://dog.ceo/api/breeds/image/random" | jq -r '.message')

    if [ -z "$img_url" ] || [ "$img_url" = "null" ]; then
        echo "  ⚠ Intento $attempts: no se obtuvo URL, reintentando..."
        sleep 1
        continue
    fi

    # Extraer extensión real del archivo
    ext="${img_url##*.}"
    ext="${ext%%\?*}"  # quitar query params si los hubiera
    [ -z "$ext" ] && ext="jpg"

    output="data/dogs/dog_$(printf '%03d' $downloaded).$ext"

    if curl -sfL -o "$output" "$img_url"; then
        ((downloaded++))
        echo "  ✓ Perro $downloaded/100"
    else
        echo "  ✗ Error descargando $img_url"
        rm -f "$output"
    fi

    sleep 0.3
done

echo "Perros descargados: $downloaded/100"

# ─── GATOS (The Cat API) ────────────────────────────────────────────
echo ""
echo "Descargando 100 imágenes de gatos..."

downloaded=0
attempts=0

while [ $downloaded -lt 100 ] && [ $attempts -lt 200 ]; do
    ((attempts++))

    # The Cat API devuelve un array JSON con url, width, height
    img_url=$(curl -sf "https://api.thecatapi.com/v1/images/search" | jq -r '.[0].url')

    if [ -z "$img_url" ] || [ "$img_url" = "null" ]; then
        echo "  ⚠ Intento $attempts: no se obtuvo URL, reintentando..."
        sleep 1
        continue
    fi

    ext="${img_url##*.}"
    ext="${ext%%\?*}"
    [ -z "$ext" ] && ext="jpg"

    output="data/cats/cat_$(printf '%03d' $downloaded).$ext"

    if curl -sfL -o "$output" "$img_url"; then
        ((downloaded++))
        echo "  ✓ Gato $downloaded/100"
    else
        echo "  ✗ Error descargando $img_url"
        rm -f "$output"
    fi

    sleep 0.3
done

echo "Gatos descargados: $downloaded/100"
echo ""
echo "¡Listo! Imágenes en data/dogs/ y data/cats/"