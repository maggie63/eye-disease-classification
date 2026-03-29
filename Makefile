# Makefile — Eye Disease Classification Pipeline
# ------------------------------------------------
# Runs the full analysis and renders the Quarto report.
#
# Usage:
#   make all        — run everything end to end
#   make data       — split raw images into train/val CSVs
#   make train      — train the model
#   make evaluate   — evaluate model and write metrics
#   make figures    — generate all figures
#   make report     — render the Quarto HTML report
#   make clean      — remove all generated outputs (keeps raw data)
#
# Prerequisites:
#   conda activate EyeClassificationPP
#   quarto installed  (https://quarto.org)
#   pip install jupyter  (needed for quarto to run python chunks)

PYTHON      = python
QUARTO = C:/PROGRA~1/Quarto/bin/quarto.cmd

# Directories
DATA_RAW    = data/raw
TABLES      = results/tables
FIGURES     = results/figures
MODELS      = results/models
REPORTS     = reports

# Key file targets (Make checks these to skip up-to-date steps)
SPLIT_CSV        = $(TABLES)/image_split.csv
SUMMARY_CSV      = $(TABLES)/dataset_summary.csv
HISTORY_CSV      = $(TABLES)/training_history.csv
PERFORMANCE_CSV  = $(TABLES)/model_performance.csv
CONFUSION_CSV    = $(TABLES)/confusion_matrix.csv
MODEL_PTH        = $(MODELS)/model.pth
TRAINING_CURVES  = $(FIGURES)/training_curves.png
CONFUSION_PNG    = $(FIGURES)/confusion_matrix.png
DIST_PNG         = $(FIGURES)/class_distribution.png
REPORT_HTML      = $(REPORTS)/eye_classifier.html


# ===========================================================================
# Top-level targets
# ===========================================================================

.PHONY: all data train evaluate figures report clean help

all: report

data: $(SPLIT_CSV)

train: $(MODEL_PTH)

evaluate: $(PERFORMANCE_CSV)

figures: $(TRAINING_CURVES) $(CONFUSION_PNG) $(DIST_PNG)

report: $(REPORT_HTML)


# ===========================================================================
# Step 1 — Split raw data
# ===========================================================================

$(SPLIT_CSV) $(SUMMARY_CSV): src/load_data.py $(DATA_RAW)
	$(PYTHON) src/load_data.py \
		--data_dir   $(DATA_RAW) \
		--output_dir $(TABLES) \
		--val_split  0.2 \
		--seed       42


# ===========================================================================
# Step 2 — Train model
# ===========================================================================

$(MODEL_PTH) $(HISTORY_CSV): src/train_model.py $(SPLIT_CSV)
	$(PYTHON) src/train_model.py \
		--split_csv   $(SPLIT_CSV) \
		--output_dir  results \
		--num_epochs  25 \
		--batch_size  4 \
		--lr          0.001 \
		--seed        42


# ===========================================================================
# Step 3 — Evaluate model
# ===========================================================================

$(PERFORMANCE_CSV) $(CONFUSION_CSV): src/evaluate_model.py $(MODEL_PTH) $(SPLIT_CSV)
	$(PYTHON) src/evaluate_model.py \
		--split_csv   $(SPLIT_CSV) \
		--model_path  $(MODEL_PTH) \
		--output_dir  $(TABLES) \
		--batch_size  4


# ===========================================================================
# Step 4 — Generate figures
# ===========================================================================

$(TRAINING_CURVES) $(CONFUSION_PNG) $(DIST_PNG): src/plot_results.py \
		$(HISTORY_CSV) $(CONFUSION_CSV) $(SUMMARY_CSV)
	$(PYTHON) src/plot_results.py \
		--tables_dir  $(TABLES) \
		--figures_dir $(FIGURES)


# ===========================================================================
# Step 5 — Render Quarto report
# ===========================================================================

$(REPORT_HTML): reports/eye_classifier.qmd \
		reports/references.bib \
		$(TRAINING_CURVES) $(CONFUSION_PNG) $(DIST_PNG) \
		$(PERFORMANCE_CSV) $(SUMMARY_CSV)
	$(QUARTO) render reports/eye_classifier.qmd


# ===========================================================================
# Clean — uses Python so it works on both Windows and Mac/Linux
# ===========================================================================

clean:
	$(PYTHON) -c "import shutil, os; \
		[shutil.rmtree(p, ignore_errors=True) for p in ['results/tables','results/figures','results/models']]; \
		os.remove('$(REPORT_HTML)') if os.path.exists('$(REPORT_HTML)') else None"

help:
	@echo ""
	@echo "Available targets:"
	@echo "  make all       — full pipeline + report"
	@echo "  make data      — split raw images into train/val CSVs"
	@echo "  make train     — train the ResNet-50 model"
	@echo "  make evaluate  — evaluate model, write metrics + confusion matrix"
	@echo "  make figures   — generate all result figures"
	@echo "  make report    — render the Quarto HTML report"
	@echo "  make clean     — remove all generated files"
	@echo ""