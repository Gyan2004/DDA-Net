#!/bin/bash
# Quick Start Guide for Medical Image Segmentation Pipeline

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Medical Image Segmentation - Kvasir-SEG                     ║"
echo "║  U-Net | DoubleU-Net | DDANet                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if dependencies are installed
echo "[1/3] Checking dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"
echo ""

# Quick demo (optional)
echo "[2/3] Run quick demo? (y/n)"
read -r demo
if [ "$demo" = "y" ]; then
    echo "Starting 5-epoch demo..."
    python quick_demo.py
    echo ""
fi

# Train models
echo "[3/3] Start training? (y/n)"
read -r train
if [ "$train" = "y" ]; then
    echo ""
    echo "Training options:"
    echo "  1) All models (U-Net, DoubleU-Net, DDANet)"
    echo "  2) U-Net only"
    echo "  3) DoubleU-Net only"
    echo "  4) DDANet only"
    echo "  5) Custom settings"
    echo ""
    read -p "Select (1-5): " option
    
    case $option in
        1)
            echo "Training all models (100 epochs)..."
            python train.py --epochs 100 --batch-size 8 --model all --loss bce_dice
            ;;
        2)
            echo "Training U-Net (100 epochs)..."
            python train.py --epochs 100 --batch-size 8 --model unet --loss bce_dice
            ;;
        3)
            echo "Training DoubleU-Net (100 epochs)..."
            python train.py --epochs 100 --batch-size 8 --model doubleunet --loss bce_dice
            ;;
        4)
            echo "Training DDANet (100 epochs)..."
            python train.py --epochs 100 --batch-size 8 --model ddanet --loss bce_dice
            ;;
        5)
            read -p "Enter epochs (default 100): " epochs
            read -p "Enter batch-size (default 8): " batch_size
            read -p "Enter learning rate (default 1e-3): " lr
            python train.py --epochs ${epochs:-100} --batch-size ${batch_size:-8} --lr ${lr:-1e-3} --model all
            ;;
    esac
    
    echo ""
    echo "Training completed!"
    echo "Evaluating models..."
    python evaluate_models.py
    
    echo ""
    echo "✓ Results saved to:"
    echo "  - checkpoints/     (trained models)"
    echo "  - results/         (evaluation plots & comparison)"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  For more options, run:                                       ║"
echo "║    python train.py --help                                     ║"
echo "║  Read documentation:                                          ║"
echo "║    README_SETUP.md           (quick start)                    ║"
echo "║    COMPLETE_PIPELINE.md      (detailed guide)                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
