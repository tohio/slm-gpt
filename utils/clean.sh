#!/usr/bin/env bash

# Exit immediately if a pipeline returns non-zero status
set -eo pipefail

print_header() {
    echo -e "\n\033[1;34m=== $1 ===\033[0m"
}

print_success() {
    echo -e "\033[0;32m✓ $1\033[0m"
}

print_warn() {
    echo -e "\033[0;33m! $1\033[0m"
}

show_help() {
    echo "Usage: ./clean.sh [OPTION]"
    echo "Options:"
    echo "  --pycache    Remove Python __pycache__ and .pyc files"
    echo "  --checkpoints Remove saved model weights in checkpoints/"
    echo "  --data       Remove all compiled binary dataset files (data/processed/)"
    echo "  --all        Full wipe: pycache, checkpoints, and processed datasets"
    echo "  --help       Show this help message"
    echo ""
    echo "Running with no arguments enters interactive selection mode."
}

clean_pycache() {
    print_header "Cleaning Python Bytecode & OS Artifacts"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
    find . -type f -name ".DS_Store" -delete 2>/dev/null || true
    print_success "Bytecode and OS files removed."
}

clean_checkpoints() {
    print_header "Cleaning Model Checkpoints"
    if [ -d "checkpoints" ]; then
        rm -rf checkpoints/*
        print_success "checkpoints/ emptied."
    else
        print_warn "No checkpoints directory found."
    fi
}

clean_data() {
    print_header "Cleaning Processed Binary Datasets"
    if [ -d "data/processed" ]; then
        rm -rf data/processed/*
        print_success "data/processed/ emptied."
    else
        print_warn "No data/processed directory found."
    fi
}

# Non-interactive CLI flags
case "$1" in
    --pycache)
        clean_pycache
        exit 0
        ;;
    --checkpoints)
        clean_checkpoints
        exit 0
        ;;
    --data)
        clean_data
        exit 0
        ;;
    --all)
        clean_pycache
        clean_checkpoints
        clean_data
        print_success "Full project cleanup complete."
        exit 0
        ;;
    --help)
        show_help
        exit 0
        ;;
esac

# Interactive prompt if no arguments provided
print_header "SLM Project Maintenance Utility"
echo "Select cleanup action:"
echo " 1) Pycache & OS junk only (safe)"
echo " 2) Checkpoints only"
echo " 3) Processed .bin data only"
echo " 4) Full reset (Pycache + Checkpoints + Processed Data)"
echo " 5) Cancel"
echo ""
read -rp "Enter choice [1-5]: " choice

case "$choice" in
    1) clean_pycache ;;
    2) clean_checkpoints ;;
    3) clean_data ;;
    4)
        read -rp "Are you sure you want to delete ALL checkpoints and compiled data? (y/N): " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            clean_pycache
            clean_checkpoints
            clean_data
            print_success "Full cleanup completed."
        else
            print_warn "Action aborted."
        fi
        ;;
    *) echo "Cancelled." ;;
esac
