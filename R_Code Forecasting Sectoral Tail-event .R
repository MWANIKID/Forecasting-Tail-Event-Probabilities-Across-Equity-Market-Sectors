# ==============================================================================
# FORECASTING SECTORAL CRASH RISK AND CONTAGION
# R PHASE: DATA PREPARATION, SECTOR CONSTRUCTION, DYNAMIC VOLATILITY,
#          EXTREME-VALUE CRASH IDENTIFICATION, AND PYTHON HANDOFF
#
# Working title:
# "Forecasting Sectoral Crash Risk and Contagion:
#  Integrating Dynamic Volatility, Extreme-Value Modelling and Graph Deep Learning"
#
# Design principle:
#   R is used once to construct the econometric/statistical foundation.
#   The output is then frozen and handed to Python/Colab for Hawkes contagion,
#   ML/DL/Graph models, survival forecasting, model evaluation, figures, and
#   final manuscript tables.
#
# IMPORTANT:
#   - No external market/macro data are used.
#   - Technical indicators are NOT used as the central feature set.
#   - Regime labels are NOT used.
#   - Crash labels are generated ex ante using rolling GARCH + EVT information.
#   - Corporate-action and thin-trading diagnostics are explicitly retained.
#
# Main method references:
#   Bollerslev, T. (1986). GARCH. Journal of Econometrics, 31(3), 307-327.
#     https://doi.org/10.1016/0304-4076(86)90063-1
#   Nelson, D.B. (1991). Conditional heteroskedasticity in asset returns:
#     A new approach. Econometrica, 59(2), 347-370.
#     https://doi.org/10.2307/2938260
#   Glosten, L.R., Jagannathan, R., & Runkle, D.E. (1993).
#     On the relation between expected value and volatility of nominal excess
#     return on stocks. Journal of Finance, 48(5), 1779-1801.
#     https://doi.org/10.1111/j.1540-6261.1993.tb05128.x
#   Pickands, J. (1975). Statistical inference using extreme order statistics.
#     Annals of Statistics, 3(1), 119-131.
#     https://doi.org/10.1214/aos/1176343003
#   Davison, A.C., & Smith, R.L. (1990). Models for exceedances over high
#     thresholds. Journal of the Royal Statistical Society B, 52(3), 393-442.
#     https://doi.org/10.1111/j.2517-6161.1990.tb01796.x
#   McNeil, A.J., & Frey, R. (2000). Estimation of tail-related risk measures
#     for heteroscedastic financial time series: An extreme value approach.
#     Journal of Empirical Finance, 7(3-4), 271-300.
#     https://doi.org/10.1016/S0927-5398(00)00012-8
#   Engle, R.F., & Manganelli, S. (2004). CAViaR.
#     Journal of Business & Economic Statistics, 22(4), 367-381.
#     https://doi.org/10.1198/073500104000000370
#   Gneiting, T., & Raftery, A.E. (2007). Strictly proper scoring rules,
#     prediction, and estimation. JASA, 102(477), 359-378.
#     https://doi.org/10.1198/016214506000001437
#   Patton, A.J. (2011). Volatility forecast comparison using imperfect
#     volatility proxies. Journal of Econometrics, 160(1), 246-256.
#     https://doi.org/10.1016/j.jeconom.2010.03.034
#
# Version: 3.4 — final audited reviewer-revision code; defensive allocation, diagnostics, and output guards
# ==============================================================================

# ==============================================================================
# 0. USER CONFIGURATION
# ==============================================================================

DATA_DIR <- Sys.getenv(
  "NSE_DATA_DIR",
  unset = "C:/..DATA From Office LapTop/PhD Research Paper/Data/Testing-Data Files/R-Data Import"
)

INPUT_FILE_NAME <- "Final Master File_All variables 01082016-31072026.csv"

PROJECT_NAME <- "Sectoral_Crash_Risk_Contagion"
OUTPUT_ROOT <- file.path(DATA_DIR, paste0(PROJECT_NAME, "_R_Outputs_v3_ReviewerRevision"))

# Reproducibility
RANDOM_SEED <- 20260831L
set.seed(RANDOM_SEED)

# Main empirical choices. These are written to the metadata file so that the
# Python phase uses exactly the same design.
TRADING_DAYS_PER_YEAR <- 252L

# Sector eligibility for PRIMARY graph/deep-learning analysis.
# All 11 sectors are retained in the exported dataset and descriptive outputs.
# A sector-day is eligible for the core model only when it is genuinely
# sector-representative rather than driven by one isolated stock.
PRIMARY_MIN_STOCKS <- 3L
PRIMARY_MIN_DAILY_RETURNS <- 2L
PRIMARY_MIN_DAILY_RETURN_SHARE <- 2 / 3
PRIMARY_MIN_ELIGIBLE_DAY_COVERAGE <- 0.90

# Corporate-action rule:
# Any explicit Bonus issue entry OR >1% change in issued shares is flagged.
SHARE_CHANGE_TOL <- 0.01

# R GARCH phase
RUN_GARCH_EVT <- TRUE
GARCH_MAIN_MODEL <- "gjrGARCH"
GARCH_MAIN_DIST <- "std"
GARCH_MIN_HISTORY <- 500L
GARCH_REFIT_EVERY <- 22L

# Candidate models are diagnostic/SI comparisons only.
GARCH_CANDIDATES <- c("sGARCH", "eGARCH", "gjrGARCH")
GARCH_CANDIDATE_DIST <- "std"

# EVT configuration.
# We model L_t = -z_t, where z_t is the standardized GARCH innovation.
EVT_THRESHOLD_PROB <- 0.90
EVT_TAIL_PROBS <- c(0.01, 0.025, 0.05)
EVT_MAIN_TAIL_PROB <- 0.025
EVT_REFIT_EVERY <- 22L
EVT_MIN_EXCEEDANCES <- 40L
EVT_SHAPE_LOWER <- -0.49
EVT_SHAPE_UPPER <- 1.50


# Reviewer-revision diagnostics. These additions DO NOT redefine the primary
# market-cap-weighted GJR-GARCH/EVT tail-event labels. They add diagnostics,
# robustness series, and an explicit market-wide common-shock series.
RUN_REVIEWER_DIAGNOSTICS <- TRUE
RUN_EQUAL_WEIGHT_ROBUSTNESS <- TRUE
RUN_STRICT_LIQUIDITY_ROBUSTNESS <- TRUE
RUN_MARKET_COMMON_SHOCK_SERIES <- TRUE

# Threshold-stability / mean-residual-life grid. Diagnostics are evaluated on
# an expanding history, matching the primary EVT design. To control runtime,
# full threshold-grid fits are evaluated every fourth successful primary refit,
# plus the first and last successful refit in each sector.
EVT_DIAGNOSTIC_THRESHOLD_PROBS <- c(0.85, 0.875, 0.90, 0.925, 0.95)
EVT_DIAGNOSTIC_REFIT_STRIDE <- 4L
EVT_CONF_LEVEL <- 0.95

# Strict-liquidity robustness: recompute the sector portfolio using only firm
# returns with positive trading volume and a non-zero daily return. The same
# minimum constituent-count and constituent-share safeguards are then applied.
ROBUST_REQUIRE_POSITIVE_VOLUME <- TRUE
ROBUST_EXCLUDE_ZERO_RETURN <- TRUE

# Source/licence notes cannot be inferred from the CSV itself. Set these
# environment variables before the final archival run so the metadata package
# records the exact provenance used in the manuscript.
DATA_SOURCE_NOTE <- Sys.getenv("NSE_DATA_SOURCE_NOTE", unset = "TO_BE_CONFIRMED")
DATA_LICENSE_NOTE <- Sys.getenv("NSE_DATA_LICENSE_NOTE", unset = "TO_BE_CONFIRMED")

# Optional automatic comparison against the previously frozen v2 handoff. If
# the old handoff exists, the script stops if the PRIMARY market-cap-weighted
# sector return or primary 2.5% event label changes unexpectedly.
REFERENCE_PRIMARY_HANDOFF <- Sys.getenv(
  "NSE_REFERENCE_PRIMARY_HANDOFF",
  unset = file.path(
    DATA_DIR,
    paste0(PROJECT_NAME, "_R_Outputs_v2"),
    "04_Python_Handoff",
    "sector_forecasting_master.csv.gz"
  )
)

# Proposed temporal split flags for Python. These can be reviewed AFTER the
# event-count feasibility table is produced, but MUST be frozen before Python
# model tuning begins.
TRAIN_END <- as.Date("2022-07-31")
VALID_END <- as.Date("2024-07-31")
# Test runs from VALID_END + 1 day to end of available data.


# Reviewer-revision temporal design established by the pre-rerun audit.
# The legacy Split column remains Train / Validation / Test for backward
# compatibility. A separate RevisionPhase column is added later:
#   TrainFit   : 2016-08-01 to 2021-01-29
#   TrainTune  : 2021-02-01 to 2022-07-29
#   Calibration: 2022-08-01 to 2024-07-31
#   Test       : 2024-08-01 to 2026-07-31
INNER_TUNE_START <- as.Date("2021-02-01")

# Multi-horizon targets will be constructed in Python from the one-day crash
# event sequence. These are stored here as metadata.
FORECAST_HORIZONS <- c(1L, 5L, 10L, 22L)

# ==============================================================================
# 1. DIRECTORY STRUCTURE
# ==============================================================================

if (!dir.exists(DATA_DIR)) {
  stop(
    paste0(
      "DATA_DIR does not exist:\n", DATA_DIR,
      "\n\nEither create the directory or set the NSE_DATA_DIR environment variable."
    )
  )
}

setwd(DATA_DIR)

DIR_TABLES <- file.path(OUTPUT_ROOT, "01_Tables")
DIR_FIGURES <- file.path(OUTPUT_ROOT, "02_Figures")
DIR_EXCEL <- file.path(OUTPUT_ROOT, "03_Excel")
DIR_HANDOFF <- file.path(OUTPUT_ROOT, "04_Python_Handoff")
DIR_LOGS <- file.path(OUTPUT_ROOT, "05_Logs")
DIR_ZIP <- file.path(OUTPUT_ROOT, "06_Zip")

for (d in c(
  OUTPUT_ROOT, DIR_TABLES, DIR_FIGURES, DIR_EXCEL,
  DIR_HANDOFF, DIR_LOGS, DIR_ZIP
)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

RUN_STAMP <- format(Sys.time(), "%Y%m%d_%H%M%S")
LOG_FILE <- file.path(DIR_LOGS, paste0("R_phase_log_", RUN_STAMP, ".txt"))

log_msg <- function(...) {
  txt <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste(..., collapse = " "))
  cat(txt, "\n")
  cat(txt, "\n", file = LOG_FILE, append = TRUE)
}

log_msg("Project:", PROJECT_NAME)
log_msg("DATA_DIR:", DATA_DIR)
log_msg("OUTPUT_ROOT:", OUTPUT_ROOT)

# ==============================================================================
# 2. PACKAGES
# ==============================================================================

REQUIRED_PACKAGES <- c(
  "data.table",
  "dplyr",
  "tidyr",
  "lubridate",
  "ggplot2",
  "scales",
  "openxlsx",
  "rugarch",
  "xts",
  "zoo",
  "FinTS",
  "moments",
  "ismev",
  "zip"
)

install_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0L) {
    log_msg("Installing missing R packages:", paste(missing, collapse = ", "))
    install.packages(missing, repos = "https://cloud.r-project.org", dependencies = TRUE)
  }
  still_missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(still_missing) > 0L) {
    stop(
      "The following required packages could not be installed: ",
      paste(still_missing, collapse = ", ")
    )
  }
}

install_if_missing(REQUIRED_PACKAGES)

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(lubridate)
  library(ggplot2)
  library(scales)
  library(openxlsx)
  library(rugarch)
  library(xts)
  library(zoo)
  library(FinTS)
  library(moments)
  library(ismev)
})

# Arrow is optional. If unavailable, the script still writes CSV.GZ handoff
# files, which Python/Colab can read directly.
ARROW_AVAILABLE <- requireNamespace("arrow", quietly = TRUE)
if (!ARROW_AVAILABLE) {
  log_msg("Optional package 'arrow' not found. Parquet export will be skipped; CSV.GZ will still be written.")
}

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

clean_numeric <- function(x) {
  x <- trimws(as.character(x))
  x[x %in% c("", "NA", "N/A", "NULL", "null", "-", ".")] <- NA_character_
  suppressWarnings(as.numeric(gsub(",", "", x, fixed = TRUE)))
}

safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  mean(x)
}

safe_sd <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 2L) return(NA_real_)
  sd(x)
}

safe_min <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  min(x)
}

safe_max <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  max(x)
}

safe_weighted_mean <- function(x, w) {
  ok <- is.finite(x) & is.finite(w) & w > 0
  if (!any(ok)) return(NA_real_)
  sum(x[ok] * w[ok]) / sum(w[ok])
}

safe_skew <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 4L) return(NA_real_)
  moments::skewness(x)
}

safe_kurt <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 4L) return(NA_real_)
  moments::kurtosis(x)
}

write_table_csv <- function(x, filename) {
  data.table::fwrite(as.data.table(x), file.path(DIR_TABLES, filename), na = "")
}

write_handoff <- function(x, basename_no_ext) {
  x_dt <- as.data.table(x)
  csv_gz <- file.path(DIR_HANDOFF, paste0(basename_no_ext, ".csv.gz"))
  data.table::fwrite(x_dt, csv_gz, na = "")
  if (ARROW_AVAILABLE) {
    parquet_file <- file.path(DIR_HANDOFF, paste0(basename_no_ext, ".parquet"))
    arrow::write_parquet(as.data.frame(x_dt), parquet_file, compression = "zstd")
  }
}

save_plot_both <- function(plot_obj, basename_no_ext, width = 11, height = 7, dpi = 320) {
  ggsave(
    filename = file.path(DIR_FIGURES, paste0(basename_no_ext, ".png")),
    plot = plot_obj, width = width, height = height, units = "in", dpi = dpi
  )
  ggsave(
    filename = file.path(DIR_FIGURES, paste0(basename_no_ext, ".pdf")),
    plot = plot_obj, width = width, height = height, units = "in"
  )
}

add_excel_sheet <- function(wb, sheet_name, x) {
  # Excel sheet names are limited to 31 characters.
  sheet_name <- substr(gsub("[\\[\\]\\*\\?/\\\\:]", "_", sheet_name), 1, 31)
  openxlsx::addWorksheet(wb, sheet_name)
  
  x_df <- as.data.frame(x)
  if (ncol(x_df) == 0L) {
    x_df <- data.frame(Note = "No rows/columns were produced for this diagnostic.")
  }
  
  openxlsx::writeDataTable(
    wb, sheet = sheet_name, x = x_df,
    tableStyle = "TableStyleMedium2", withFilter = TRUE
  )
  openxlsx::freezePane(wb, sheet = sheet_name, firstRow = TRUE)
  openxlsx::setColWidths(
    wb, sheet = sheet_name, cols = seq_len(ncol(x_df)), widths = "auto"
  )
}

first_existing_col <- function(df, candidates) {
  nm <- names(df)
  hit <- candidates[candidates %in% nm]
  if (length(hit) == 0L) return(NA_character_)
  hit[1]
}


tail_suffix <- function(a) {
  if (abs(a - 0.01) < 1e-12) return("001")
  if (abs(a - 0.025) < 1e-12) return("0025")
  if (abs(a - 0.05) < 1e-12) return("005")
  # Generic fallback for any future user-specified probability.
  paste0("p", gsub("\\.", "_", format(a, scientific = FALSE, trim = TRUE)))
}

# Generalized Pareto likelihood for excesses y > 0.
# Parameters are (log(scale), shape). The log-scale parameterization enforces
# scale > 0, while the support condition 1 + shape*y/scale > 0 is checked
# explicitly. This function is also used to validate estimates returned by
# external and internal optimizers.
gpd_negloglik <- function(par, y) {
  beta <- exp(par[1])
  xi <- par[2]
  
  if (
    !is.finite(beta) || beta <= 0 ||
    !is.finite(xi) ||
    xi < EVT_SHAPE_LOWER || xi > EVT_SHAPE_UPPER
  ) {
    return(1e30)
  }
  
  n <- length(y)
  
  if (abs(xi) < 1e-8) {
    val <- n * log(beta) + sum(y / beta)
  } else {
    arg <- 1 + xi * y / beta
    if (any(!is.finite(arg)) || any(arg <= 0)) return(1e30)
    val <- n * log(beta) + (1 / xi + 1) * sum(log(arg))
  }
  
  if (!is.finite(val)) 1e30 else val
}

# Independent multi-start base-R optimizer. This serves both as a robust
# estimator and as an independent cross-check on ismev::gpd.fit().
fit_gpd_nlminb <- function(y) {
  scale0 <- max(mean(y), stats::sd(y), 1e-6)
  scale_factors <- c(0.50, 1.00, 2.00)
  shape_starts <- c(-0.25, -0.10, 0.05, 0.10, 0.25, 0.50)
  upper_scale <- log(max(c(max(y) * 100, scale0 * 100, 10), na.rm = TRUE))
  
  fits <- list()
  
  for (sf in scale_factors) {
    for (xi0 in shape_starts) {
      start_par <- c(log(max(scale0 * sf, 1e-8)), xi0)
      
      fit <- tryCatch(
        stats::nlminb(
          start = start_par,
          objective = gpd_negloglik,
          y = y,
          lower = c(log(1e-10), EVT_SHAPE_LOWER),
          upper = c(upper_scale, EVT_SHAPE_UPPER),
          control = list(
            iter.max = 5000,
            eval.max = 10000,
            rel.tol = 1e-10,
            x.tol = 1e-10
          )
        ),
        error = function(e) NULL
      )
      
      if (!is.null(fit) && is.finite(fit$objective)) {
        beta <- exp(fit$par[1])
        xi <- fit$par[2]
        support_ok <- beta > 0 && all(1 + xi * y / beta > 0)
        
        if (support_ok) {
          fits[[length(fits) + 1L]] <- data.table(
            Method = "nlminb_multistart",
            Scale = beta,
            Shape = xi,
            NLL = fit$objective,
            Convergence = as.integer(fit$convergence)
          )
        }
      }
    }
  }
  
  if (length(fits) == 0L) return(NULL)
  
  tab <- rbindlist(fits, use.names = TRUE, fill = TRUE)
  # Prefer formally converged candidates. If none converged, retain the best
  # finite candidate but mark it non-zero so the caller can reject it.
  conv <- tab[Convergence == 0L]
  if (nrow(conv) > 0L) {
    conv[which.min(NLL)]
  } else {
    tab[which.min(NLL)]
  }
}

# Robust GPD MLE for exceedances y > 0.
#
# The previous Version 1.0 optimizer could report convergence without moving
# away from its starting values. Version 2.1 therefore:
#   (i) fits the GPD with the established ismev implementation,
#  (ii) independently fits it with multi-start nlminb,
# (iii) evaluates every candidate with the SAME explicit log-likelihood,
#  (iv) chooses the valid candidate with the highest likelihood, and
#   (v) records the fitting method and objective improvement for audit.
fit_gpd_mle <- function(excess) {
  y <- excess[is.finite(excess) & excess > 0]
  n <- length(y)
  
  if (n < EVT_MIN_EXCEEDANCES || length(unique(y)) < 5L) {
    return(list(
      scale = NA_real_, shape = NA_real_, convergence = 99L,
      n_exceed = n, loglik = NA_real_, method = "insufficient_data",
      nll_improvement = NA_real_, at_boundary = NA
    ))
  }
  
  scale_start <- max(mean(y), stats::sd(y), 1e-6)
  reference_nll <- gpd_negloglik(c(log(scale_start), 0.10), y)
  
  candidates <- list()
  
  # Established EVT implementation.
  fit_ismev <- tryCatch(
    suppressWarnings(
      ismev::gpd.fit(
        y,
        threshold = 0,
        show = FALSE
      )
    ),
    error = function(e) NULL
  )
  
  if (
    !is.null(fit_ismev) &&
    !is.null(fit_ismev$mle) &&
    length(fit_ismev$mle) >= 2L
  ) {
    beta_i <- as.numeric(fit_ismev$mle[1])
    xi_i <- as.numeric(fit_ismev$mle[2])
    
    if (
      is.finite(beta_i) && beta_i > 0 &&
      is.finite(xi_i) &&
      xi_i >= EVT_SHAPE_LOWER && xi_i <= EVT_SHAPE_UPPER &&
      all(1 + xi_i * y / beta_i > 0)
    ) {
      nll_i <- gpd_negloglik(c(log(beta_i), xi_i), y)
      if (is.finite(nll_i) && nll_i < 1e29) {
        candidates[[length(candidates) + 1L]] <- data.table(
          Method = "ismev_gpd.fit",
          Scale = beta_i,
          Shape = xi_i,
          NLL = nll_i,
          Convergence = 0L
        )
      }
    }
  }
  
  # Independent multi-start likelihood fit.
  fit_nl <- fit_gpd_nlminb(y)
  if (!is.null(fit_nl) && nrow(fit_nl) == 1L) {
    candidates[[length(candidates) + 1L]] <- fit_nl
  }
  
  if (length(candidates) == 0L) {
    return(list(
      scale = NA_real_, shape = NA_real_, convergence = 98L,
      n_exceed = n, loglik = NA_real_, method = "all_estimators_failed",
      nll_improvement = NA_real_, at_boundary = NA
    ))
  }
  
  cand <- rbindlist(candidates, use.names = TRUE, fill = TRUE)
  cand <- cand[
    is.finite(Scale) & Scale > 0 &
      is.finite(Shape) &
      Shape >= EVT_SHAPE_LOWER & Shape <= EVT_SHAPE_UPPER &
      is.finite(NLL)
  ]
  
  if (nrow(cand) == 0L) {
    return(list(
      scale = NA_real_, shape = NA_real_, convergence = 97L,
      n_exceed = n, loglik = NA_real_, method = "no_valid_candidate",
      nll_improvement = NA_real_, at_boundary = NA
    ))
  }
  
  # Prefer converged candidates, then the maximum-likelihood solution.
  cand_conv <- cand[Convergence == 0L]
  best <- if (nrow(cand_conv) > 0L) {
    cand_conv[which.min(NLL)]
  } else {
    cand[which.min(NLL)]
  }
  
  beta_hat <- best$Scale[1]
  xi_hat <- best$Shape[1]
  best_nll <- best$NLL[1]
  
  # Final support and local-solution checks.
  support_ok <- all(1 + xi_hat * y / beta_hat > 0)
  at_boundary <- (
    abs(xi_hat - EVT_SHAPE_LOWER) < 1e-5 ||
      abs(xi_hat - EVT_SHAPE_UPPER) < 1e-5
  )
  
  if (!support_ok || !is.finite(best_nll)) {
    return(list(
      scale = NA_real_, shape = NA_real_, convergence = 96L,
      n_exceed = n, loglik = NA_real_, method = "support_failure",
      nll_improvement = NA_real_, at_boundary = at_boundary
    ))
  }
  
  list(
    scale = beta_hat,
    shape = xi_hat,
    convergence = as.integer(best$Convergence[1]),
    n_exceed = n,
    loglik = -best_nll,
    method = as.character(best$Method[1]),
    nll_improvement = reference_nll - best_nll,
    at_boundary = at_boundary
  )
}

gpd_loss_quantile <- function(u, p_u, alpha, scale, shape) {
  if (
    any(!is.finite(c(u, p_u, alpha, scale, shape))) ||
    alpha <= 0 || alpha >= p_u || p_u <= 0 || scale <= 0
  ) {
    return(NA_real_)
  }
  
  if (abs(shape) < 1e-7) {
    u + scale * log(p_u / alpha)
  } else {
    u + (scale / shape) * (((p_u / alpha)^shape) - 1)
  }
}

make_garch_spec <- function(model = GARCH_MAIN_MODEL, dist = GARCH_MAIN_DIST) {
  rugarch::ugarchspec(
    variance.model = list(
      model = model,
      garchOrder = c(1, 1)
    ),
    mean.model = list(
      armaOrder = c(0, 0),
      include.mean = TRUE
    ),
    distribution.model = dist
  )
}

fit_garch_candidate <- function(x_pct, sector_name, model_name, dist_name) {
  spec <- make_garch_spec(model = model_name, dist = dist_name)
  
  fit <- tryCatch(
    rugarch::ugarchfit(
      spec = spec,
      data = x_pct,
      solver = "hybrid",
      solver.control = list(trace = 0)
    ),
    error = function(e) NULL
  )
  
  if (is.null(fit)) {
    return(data.table(
      Sector = sector_name,
      Model = model_name,
      Distribution = dist_name,
      Convergence = 99L,
      AIC = NA_real_,
      BIC = NA_real_,
      HQIC = NA_real_,
      LB_resid_p = NA_real_,
      LB_sq_resid_p = NA_real_,
      ARCH_LM_p = NA_real_
    ))
  }
  
  ic <- rugarch::infocriteria(fit)
  z <- as.numeric(residuals(fit, standardize = TRUE))
  z <- z[is.finite(z)]
  
  lb_r <- if (length(z) > 20) Box.test(z, lag = 10, type = "Ljung-Box")$p.value else NA_real_
  lb_sq <- if (length(z) > 20) Box.test(z^2, lag = 10, type = "Ljung-Box")$p.value else NA_real_
  arch_p <- if (length(z) > 30) {
    tryCatch(FinTS::ArchTest(z, lags = 10)$p.value, error = function(e) NA_real_)
  } else {
    NA_real_
  }
  
  data.table(
    Sector = sector_name,
    Model = model_name,
    Distribution = dist_name,
    Convergence = fit@fit$convergence,
    AIC = unname(ic["Akaike"]),
    BIC = unname(ic["Bayes"]),
    HQIC = unname(ic["Hannan-Quinn"]),
    LB_resid_p = lb_r,
    LB_sq_resid_p = lb_sq,
    ARCH_LM_p = arch_p
  )
}

# Extract uGARCHroll density output robustly across rugarch versions.
extract_roll_density <- function(roll_obj) {
  out <- tryCatch(
    as.data.frame(roll_obj, which = "density"),
    error = function(e) tryCatch(as.data.frame(roll_obj), error = function(e2) NULL)
  )
  if (is.null(out)) return(NULL)
  
  out$Date <- suppressWarnings(as.Date(rownames(out)))
  if (all(is.na(out$Date))) {
    # Some versions may carry dates in another index representation.
    out$Date <- suppressWarnings(as.Date(as.character(rownames(out))))
  }
  out
}

# Rolling GARCH + rolling EVT for one sector.
run_sector_garch_evt <- function(sector_name, sector_df) {
  log_msg("GARCH/EVT started:", sector_name)
  
  dd <- as.data.table(sector_df)
  dd <- dd[
    Sector == sector_name &
      PrimarySector == TRUE &
      is.finite(SectorReturn_Model)
  ]
  setorder(dd, Date)
  
  if (nrow(dd) <= GARCH_MIN_HISTORY + 100L) {
    log_msg("Skipping", sector_name, "- too few valid sector returns:", nrow(dd))
    return(list(
      panel = data.table(),
      evt_params = data.table(),
      error = data.table(Sector = sector_name, Error = "Insufficient valid returns")
    ))
  }
  
  # Use percentage returns for numerical stability in GARCH estimation.
  x_pct <- dd$SectorReturn_Model * 100
  x_xts <- xts::xts(x_pct, order.by = dd$Date)
  
  spec <- make_garch_spec(GARCH_MAIN_MODEL, GARCH_MAIN_DIST)
  
  # Initial fit is used to seed standardized residual history for EVT.
  init_x <- x_pct[seq_len(GARCH_MIN_HISTORY)]
  init_fit <- tryCatch(
    rugarch::ugarchfit(
      spec = spec,
      data = init_x,
      solver = "hybrid",
      solver.control = list(trace = 0)
    ),
    error = function(e) NULL
  )
  
  if (is.null(init_fit) || init_fit@fit$convergence != 0L) {
    log_msg("Initial GARCH fit failed:", sector_name)
    return(list(
      panel = data.table(),
      evt_params = data.table(),
      error = data.table(Sector = sector_name, Error = "Initial GARCH fit failed")
    ))
  }
  
  z_init <- as.numeric(residuals(init_fit, standardize = TRUE))
  mu_init <- as.numeric(fitted(init_fit))
  sig_init <- as.numeric(sigma(init_fit))
  
  # Genuine one-step rolling forecasts after the initial history.
  roll_obj <- tryCatch(
    rugarch::ugarchroll(
      spec = spec,
      data = x_xts,
      n.start = GARCH_MIN_HISTORY,
      refit.every = GARCH_REFIT_EVERY,
      refit.window = "recursive",
      solver = "hybrid",
      calculate.VaR = FALSE,
      keep.coef = TRUE
    ),
    error = function(e) {
      log_msg("ugarchroll error in", sector_name, ":", conditionMessage(e))
      NULL
    }
  )
  
  if (is.null(roll_obj)) {
    return(list(
      panel = data.table(),
      evt_params = data.table(),
      error = data.table(Sector = sector_name, Error = "ugarchroll failed")
    ))
  }
  
  roll_df <- extract_roll_density(roll_obj)
  if (is.null(roll_df) || nrow(roll_df) == 0L) {
    return(list(
      panel = data.table(),
      evt_params = data.table(),
      error = data.table(Sector = sector_name, Error = "Could not extract ugarchroll density")
    ))
  }
  
  # Detect column names robustly.
  col_mu <- first_existing_col(roll_df, c("Mu", "mu"))
  col_sigma <- first_existing_col(roll_df, c("Sigma", "sigma"))
  col_real <- first_existing_col(roll_df, c("Realized", "realized"))
  
  if (any(is.na(c(col_mu, col_sigma, col_real)))) {
    return(list(
      panel = data.table(),
      evt_params = data.table(),
      error = data.table(
        Sector = sector_name,
        Error = paste0("Missing expected ugarchroll columns. Found: ", paste(names(roll_df), collapse = ", "))
      )
    ))
  }
  
  # Make an owned data.table copy before any by-reference additions.
  # Package-returned objects can otherwise retain shallow-copy attributes.
  roll_dt <- data.table::copy(data.table::as.data.table(roll_df))
  data.table::setDT(roll_dt)
  data.table::setalloccol(roll_dt, n = ncol(roll_dt) + 12L)
  
  roll_dt[, `:=`(
    MuPct = as.numeric(get(col_mu)),
    SigmaPct = as.numeric(get(col_sigma)),
    RealizedPct = as.numeric(get(col_real))
  )]
  roll_dt[, StdInnovation := (RealizedPct - MuPct) / SigmaPct]
  
  # Date extraction fallback: if row names were not parsed, align with the
  # post-initial valid dates in the same order.
  if (all(is.na(roll_dt$Date))) {
    roll_dt[, Date := dd$Date[(GARCH_MIN_HISTORY + 1L):nrow(dd)]]
  }
  
  roll_dt[, Sector := sector_name]
  
  # Build a complete standardized-residual history:
  init_dt <- data.table(
    Date = dd$Date[seq_len(GARCH_MIN_HISTORY)],
    Sector = sector_name,
    MuPct = mu_init,
    SigmaPct = sig_init,
    RealizedPct = init_x,
    StdInnovation = z_init,
    ForecastEligible = FALSE
  )
  
  roll_small <- roll_dt[, .(
    Date,
    Sector,
    MuPct,
    SigmaPct,
    RealizedPct,
    StdInnovation,
    ForecastEligible = TRUE
  )]
  
  garch_all <- rbindlist(list(init_dt, roll_small), use.names = TRUE, fill = TRUE)
  setorder(garch_all, Date)
  garch_all <- data.table::copy(garch_all[!duplicated(Date)])
  
  # Restore/pre-allocate the data.table internal column pointer before adding
  # many columns by reference. This prevents shallow-copy/setalloccol failures.
  data.table::setDT(garch_all)
  data.table::setalloccol(garch_all, n = ncol(garch_all) + 40L)
  
  required_garch_all_cols <- c(
    "Date", "Sector", "MuPct", "SigmaPct", "RealizedPct",
    "StdInnovation", "ForecastEligible"
  )
  missing_garch_all_cols <- setdiff(required_garch_all_cols, names(garch_all))
  if (length(missing_garch_all_cols) > 0L) {
    return(list(
      panel = data.table(),
      evt_params = data.table(),
      error = data.table(
        Sector = sector_name,
        Error = paste0(
          "Internal GARCH table missing required columns: ",
          paste(missing_garch_all_cols, collapse = ", ")
        )
      )
    ))
  }
  
  # Rolling EVT. Parameters are estimated only from standardized innovations
  # observed strictly BEFORE the date being labelled.
  garch_all[, `:=`(
    EVT_u = NA_real_,
    EVT_scale = NA_real_,
    EVT_shape = NA_real_,
    EVT_pu = NA_real_,
    EVT_nhist = NA_integer_,
    EVT_nexc = NA_integer_,
    EVT_fit_convergence = NA_integer_,
    EVT_fit_method = NA_character_,
    EVT_loglik = NA_real_,
    EVT_nll_improvement = NA_real_,
    EVT_at_boundary = NA,
    EVT_refit_id = NA_integer_
  )]
  
  for (a in EVT_TAIL_PROBS) {
    nm_q <- paste0("EVT_LossQ_", tail_suffix(a))
    nm_thr <- paste0("CrashThresholdPct_", tail_suffix(a))
    nm_evt <- paste0("Crash_", tail_suffix(a))
    garch_all[, (nm_q) := NA_real_]
    garch_all[, (nm_thr) := NA_real_]
    garch_all[, (nm_evt) := NA_integer_]
  }
  
  evt_param_rows <- list()
  current_fit <- NULL
  current_refit_id <- 0L
  
  start_i <- GARCH_MIN_HISTORY + 1L
  
  for (i in seq.int(start_i, nrow(garch_all))) {
    need_refit <- is.null(current_fit) ||
      ((i - start_i) %% EVT_REFIT_EVERY == 0L)
    
    if (need_refit) {
      z_hist <- garch_all$StdInnovation[seq_len(i - 1L)]
      z_hist <- z_hist[is.finite(z_hist)]
      
      # Loss = - standardized return shock; large values are adverse shocks.
      loss_hist <- -z_hist
      
      if (length(loss_hist) >= GARCH_MIN_HISTORY) {
        u <- as.numeric(
          stats::quantile(
            loss_hist,
            probs = EVT_THRESHOLD_PROB,
            na.rm = TRUE,
            names = FALSE,
            type = 8
          )
        )
        excess <- loss_hist[loss_hist > u] - u
        p_u <- length(excess) / length(loss_hist)
        
        fit <- fit_gpd_mle(excess)
        
        if (
          is.finite(fit$scale) &&
          is.finite(fit$shape) &&
          fit$convergence == 0L &&
          fit$n_exceed >= EVT_MIN_EXCEEDANCES
        ) {
          current_refit_id <- current_refit_id + 1L
          current_fit <- list(
            u = u,
            scale = fit$scale,
            shape = fit$shape,
            p_u = p_u,
            n_hist = length(loss_hist),
            n_exc = fit$n_exceed,
            convergence = fit$convergence,
            method = fit$method,
            loglik = fit$loglik,
            nll_improvement = fit$nll_improvement,
            at_boundary = fit$at_boundary,
            refit_id = current_refit_id,
            refit_date = garch_all$Date[i]
          )
          
          evt_param_rows[[length(evt_param_rows) + 1L]] <- data.table(
            Sector = sector_name,
            RefitID = current_refit_id,
            RefitDate = garch_all$Date[i],
            HistN = length(loss_hist),
            ThresholdProb = EVT_THRESHOLD_PROB,
            Threshold_u = u,
            ExceedN = fit$n_exceed,
            ExceedRate = p_u,
            GPD_Scale = fit$scale,
            GPD_Shape = fit$shape,
            GPD_LogLik = fit$loglik,
            FitMethod = fit$method,
            NLL_Improvement = fit$nll_improvement,
            AtBoundary = fit$at_boundary,
            Convergence = fit$convergence
          )
        }
      }
    }
    
    if (!is.null(current_fit)) {
      garch_all$EVT_u[i] <- current_fit$u
      garch_all$EVT_scale[i] <- current_fit$scale
      garch_all$EVT_shape[i] <- current_fit$shape
      garch_all$EVT_pu[i] <- current_fit$p_u
      garch_all$EVT_nhist[i] <- current_fit$n_hist
      garch_all$EVT_nexc[i] <- current_fit$n_exc
      garch_all$EVT_fit_convergence[i] <- current_fit$convergence
      garch_all$EVT_fit_method[i] <- current_fit$method
      garch_all$EVT_loglik[i] <- current_fit$loglik
      garch_all$EVT_nll_improvement[i] <- current_fit$nll_improvement
      garch_all$EVT_at_boundary[i] <- current_fit$at_boundary
      garch_all$EVT_refit_id[i] <- current_fit$refit_id
      
      for (a in EVT_TAIL_PROBS) {
        suffix <- tail_suffix(a)
        nm_q <- paste0("EVT_LossQ_", suffix)
        nm_thr <- paste0("CrashThresholdPct_", suffix)
        nm_evt <- paste0("Crash_", suffix)
        
        q_loss <- gpd_loss_quantile(
          u = current_fit$u,
          p_u = current_fit$p_u,
          alpha = a,
          scale = current_fit$scale,
          shape = current_fit$shape
        )
        
        threshold_pct <- if (
          is.finite(q_loss) &&
          is.finite(garch_all$MuPct[i]) &&
          is.finite(garch_all$SigmaPct[i])
        ) {
          garch_all$MuPct[i] - garch_all$SigmaPct[i] * q_loss
        } else {
          NA_real_
        }
        
        garch_all[[nm_q]][i] <- q_loss
        garch_all[[nm_thr]][i] <- threshold_pct
        
        if (
          isTRUE(garch_all$ForecastEligible[i]) &&
          is.finite(garch_all$RealizedPct[i]) &&
          is.finite(threshold_pct)
        ) {
          garch_all[[nm_evt]][i] <- as.integer(garch_all$RealizedPct[i] < threshold_pct)
        }
      }
    }
  }
  
  evt_params <- if (length(evt_param_rows) > 0L) {
    rbindlist(evt_param_rows, use.names = TRUE, fill = TRUE)
  } else {
    data.table()
  }
  
  # Convert main GARCH outputs back to decimal units for Python consistency.
  # Reassert ownership because repeated column updates inside the EVT loop can
  # cause R to rematerialise the object.
  garch_all <- data.table::copy(garch_all)
  data.table::setDT(garch_all)
  data.table::setalloccol(garch_all, n = ncol(garch_all) + 12L)
  
  data.table::set(garch_all, j = "GARCH_Mu", value = garch_all$MuPct / 100)
  data.table::set(garch_all, j = "GARCH_Sigma", value = garch_all$SigmaPct / 100)
  data.table::set(garch_all, j = "GARCH_Realized", value = garch_all$RealizedPct / 100)
  
  # Add main tail-event aliases for convenience.
  main_suffix <- tail_suffix(EVT_MAIN_TAIL_PROB)
  main_crash_col <- paste0("Crash_", main_suffix)
  main_thr_col <- paste0("CrashThresholdPct_", main_suffix)
  
  data.table::set(garch_all, j = "Crash_Main", value = garch_all[[main_crash_col]])
  data.table::set(
    garch_all,
    j = "CrashThreshold_Main",
    value = garch_all[[main_thr_col]] / 100
  )
  
  out_panel <- garch_all[, .(
    Date,
    Sector,
    GARCH_Mu,
    GARCH_Sigma,
    # Needed by reviewer robustness comparisons of alternative sector returns.
    GARCH_Realized,
    StdInnovation,
    EVT_u,
    EVT_scale,
    EVT_shape,
    EVT_pu,
    EVT_nhist,
    EVT_nexc,
    EVT_fit_convergence,
    EVT_fit_method,
    EVT_loglik,
    EVT_nll_improvement,
    EVT_at_boundary,
    EVT_refit_id,
    CrashThreshold_001 = CrashThresholdPct_001 / 100,
    CrashThreshold_0025 = CrashThresholdPct_0025 / 100,
    CrashThreshold_005 = CrashThresholdPct_005 / 100,
    # Primary-threshold alias retained explicitly for reviewer diagnostics and
    # downstream compatibility. It is exactly the 2.5% EVT threshold.
    CrashThreshold_Main = CrashThresholdPct_0025 / 100,
    Crash_001,
    Crash_0025,
    Crash_005,
    Crash_Main
  )]
  
  log_msg(
    "GARCH/EVT completed:", sector_name,
    "| valid labelled observations:", sum(!is.na(out_panel$Crash_Main)),
    "| main crashes:", sum(out_panel$Crash_Main == 1L, na.rm = TRUE)
  )
  
  list(
    panel = out_panel,
    evt_params = evt_params,
    error = data.table()
  )
}

# ==============================================================================
# 4. LOCATE AND IMPORT THE MASTER FILE
# ==============================================================================

INPUT_FILE <- file.path(DATA_DIR, INPUT_FILE_NAME)

if (!file.exists(INPUT_FILE)) {
  candidates <- list.files(
    DATA_DIR,
    pattern = "Final Master File.*01082016.*31072026.*\\.csv$",
    full.names = TRUE,
    ignore.case = TRUE
  )
  if (length(candidates) == 1L) {
    INPUT_FILE <- candidates[1]
  } else {
    stop(
      paste0(
        "Could not uniquely locate the master CSV.\nExpected:\n",
        file.path(DATA_DIR, INPUT_FILE_NAME),
        "\nCandidates found: ", length(candidates)
      )
    )
  }
}

log_msg("Reading:", INPUT_FILE)

raw <- data.table::fread(
  INPUT_FILE,
  na.strings = c("", "NA", "N/A", "NULL", "null"),
  encoding = "UTF-8",
  showProgress = TRUE
)

# Exact expected source columns.
EXPECTED_COLUMNS <- c(
  "Date", "Open", "High", "Low", "Close", "Volume",
  "Category", "Stock", "Bonus issue", "Issued Shares",
  "Market Capitalization (KES)"
)

missing_expected <- setdiff(EXPECTED_COLUMNS, names(raw))
if (length(missing_expected) > 0L) {
  stop(
    "The master CSV is missing expected columns: ",
    paste(missing_expected, collapse = ", ")
  )
}

# Rename once to analysis-safe names.
setnames(
  raw,
  old = EXPECTED_COLUMNS,
  new = c(
    "DateRaw", "Open", "High", "Low", "Close", "Volume",
    "Sector", "Stock", "BonusIssueRaw", "IssuedShares",
    "MarketCapKES"
  )
)

raw[, SourceRow := .I]

# Parse fields.
raw[, Date := suppressWarnings(lubridate::dmy(DateRaw, quiet = TRUE))]
raw[, `:=`(
  Open = clean_numeric(Open),
  High = clean_numeric(High),
  Low = clean_numeric(Low),
  Close = clean_numeric(Close),
  Volume = clean_numeric(Volume),
  BonusIssueShares = clean_numeric(BonusIssueRaw),
  IssuedShares = clean_numeric(IssuedShares),
  MarketCapKES = clean_numeric(MarketCapKES),
  Sector = trimws(as.character(Sector)),
  Stock = trimws(as.character(Stock))
)]

# ==============================================================================
# 5. RAW-DATA AUDIT AND CONSERVATIVE CLEANING
# ==============================================================================

audit_overview <- data.table(
  Metric = c(
    "Raw rows",
    "Unique stocks",
    "Unique sectors",
    "Unique parsed dates",
    "Minimum date",
    "Maximum date",
    "Duplicate stock-date rows"
  ),
  Value = c(
    nrow(raw),
    uniqueN(raw$Stock, na.rm = TRUE),
    uniqueN(raw$Sector, na.rm = TRUE),
    uniqueN(raw$Date, na.rm = TRUE),
    as.character(min(raw$Date, na.rm = TRUE)),
    as.character(max(raw$Date, na.rm = TRUE)),
    sum(duplicated(raw[, .(Date, Stock)]), na.rm = TRUE)
  )
)

missingness_raw <- data.table(
  Variable = names(raw),
  MissingN = vapply(raw, function(x) sum(is.na(x)), integer(1)),
  MissingPct = 100 * vapply(raw, function(x) mean(is.na(x)), numeric(1))
)

raw[, CriticalMissing := (
  is.na(Date) |
    is.na(Stock) | Stock == "" |
    is.na(Sector) | Sector == "" |
    !is.finite(Close) | Close <= 0 |
    !is.finite(IssuedShares) | IssuedShares <= 0 |
    !is.finite(MarketCapKES) | MarketCapKES <= 0
)]

raw[, InvalidOHLC := !(
  is.finite(Open) & Open > 0 &
    is.finite(High) & High > 0 &
    is.finite(Low) & Low > 0 &
    is.finite(Close) & Close > 0 &
    High >= pmax(Open, Low, Close) &
    Low <= pmin(Open, High, Close)
)]

# Negative volume is invalid; zero volume is retained as economically meaningful.
raw[is.finite(Volume) & Volume < 0, Volume := NA_real_]

critical_rows <- raw[CriticalMissing == TRUE]
invalid_ohlc_rows <- raw[InvalidOHLC == TRUE & CriticalMissing == FALSE]

clean <- raw[CriticalMissing == FALSE]

# No silent handling of duplicates.
dup_stock_date <- clean[duplicated(clean[, .(Date, Stock)]) | duplicated(clean[, .(Date, Stock)], fromLast = TRUE)]
if (nrow(dup_stock_date) > 0L) {
  data.table::fwrite(
    dup_stock_date,
    file.path(DIR_TABLES, "ERROR_duplicate_stock_date_rows.csv")
  )
  stop(
    "Duplicate Stock-Date rows were found after cleaning. ",
    "They have been exported. Resolve them before continuing."
  )
}

setorder(clean, Stock, Date)

# ==============================================================================
# 6. TRADING CALENDAR, STOCK RETURNS, THIN TRADING, CORPORATE ACTIONS
# ==============================================================================

calendar <- data.table(Date = sort(unique(clean$Date)))
calendar[, TradingIndex := seq_len(.N)]

clean <- merge(clean, calendar, by = "Date", all.x = TRUE, sort = FALSE)
setorder(clean, Stock, Date)

clean[, `:=`(
  LagClose = shift(Close),
  LagMarketCapKES = shift(MarketCapKES),
  LagIssuedShares = shift(IssuedShares),
  LagTradingIndex = shift(TradingIndex)
), by = Stock]

clean[, TradingGap := TradingIndex - LagTradingIndex]
clean[, ShareRatio := IssuedShares / LagIssuedShares]

clean[, BonusFlag := !is.na(BonusIssueShares) & BonusIssueShares > 0]
clean[, ShareChangeFlag := (
  is.finite(ShareRatio) &
    abs(log(ShareRatio)) > log(1 + SHARE_CHANGE_TOL)
)]
clean[is.na(ShareChangeFlag), ShareChangeFlag := FALSE]

clean[, CorporateActionFlag := BonusFlag | ShareChangeFlag]

# ±1 observed-row window is generated as a robustness flag only.
clean[, CorporateActionWindow3 := (
  CorporateActionFlag |
    shift(CorporateActionFlag, 1L, type = "lag", fill = FALSE) |
    shift(CorporateActionFlag, 1L, type = "lead", fill = FALSE)
), by = Stock]

# Primary "daily" return is only created for consecutive MARKET trading dates.
# Returns spanning missing market sessions are left NA rather than attributed
# entirely to the next observed day.
clean[, RawLogReturn := fifelse(
  is.finite(Close) &
    is.finite(LagClose) &
    Close > 0 &
    LagClose > 0 &
    TradingGap == 1L,
  log(Close / LagClose),
  NA_real_
)]

# Remove the event-date return when a corporate action/share-count break is
# flagged. We do NOT automatically attempt a split adjustment because the Bonus
# issue field is not assumed to encode a clean split ratio.
clean[, LogReturn := fifelse(
  CorporateActionFlag,
  NA_real_,
  RawLogReturn
)]

# Robustness version excluding event date ± one adjacent observed row.
clean[, LogReturn_CA3 := fifelse(
  CorporateActionWindow3,
  NA_real_,
  RawLogReturn
)]

# Thin-trading diagnostics.
clean[, ZeroReturn := fifelse(
  is.finite(LogReturn),
  as.integer(abs(LogReturn) < 1e-12),
  NA_integer_
)]
clean[, ZeroVolume := fifelse(
  is.finite(Volume),
  as.integer(Volume == 0),
  NA_integer_
)]
clean[, NegativeReturn := fifelse(
  is.finite(LogReturn),
  as.integer(LogReturn < 0),
  NA_integer_
)]

# ==============================================================================
# 7. STOCK-LEVEL INTERNAL FEATURES FROM THE UPLOADED DATA ONLY
# ==============================================================================

# OHLC validity is used to gate range-volatility measures, not to discard the row.
clean[, OHLCValid := !InvalidOHLC]

clean[, ParkinsonVar := fifelse(
  OHLCValid,
  (log(High / Low)^2) / (4 * log(2)),
  NA_real_
)]

clean[, GarmanKlassVar := fifelse(
  OHLCValid,
  pmax(
    0,
    0.5 * (log(High / Low)^2) -
      (2 * log(2) - 1) * (log(Close / Open)^2)
  ),
  NA_real_
)]

clean[, RogersSatchellVar := fifelse(
  OHLCValid,
  pmax(
    0,
    log(High / Close) * log(High / Open) +
      log(Low / Close) * log(Low / Open)
  ),
  NA_real_
)]

clean[, Turnover := fifelse(
  is.finite(Volume) & Volume >= 0 &
    is.finite(IssuedShares) & IssuedShares > 0,
  Volume / IssuedShares,
  NA_real_
)]

clean[, TradedValueKES := fifelse(
  is.finite(Volume) & Volume > 0 &
    is.finite(Close) & Close > 0,
  Volume * Close,
  NA_real_
)]

# Scaled Amihud-style measure. Zero-volume observations remain NA here and are
# separately represented by ZeroVolume.
clean[, AmihudILLIQ := fifelse(
  is.finite(LogReturn) &
    is.finite(TradedValueKES) &
    TradedValueKES > 0,
  abs(LogReturn) / TradedValueKES * 1e8,
  NA_real_
)]

# ==============================================================================
# 8. SECTOR CONSTRUCTION
# ==============================================================================

# Lagged market capitalization is the weight to prevent contemporaneous
# look-ahead in the sector return.
sector_daily <- clean[, {
  ret_ok <- is.finite(LogReturn) & is.finite(LagMarketCapKES) & LagMarketCapKES > 0
  cap_ok <- is.finite(LagMarketCapKES) & LagMarketCapKES > 0
  
  w_ret <- if (any(ret_ok)) LagMarketCapKES[ret_ok] / sum(LagMarketCapKES[ret_ok]) else numeric(0)
  w_cap <- if (any(cap_ok)) LagMarketCapKES[cap_ok] / sum(LagMarketCapKES[cap_ok]) else numeric(0)
  
  sector_ret_mcw <- if (any(ret_ok)) sum(LogReturn[ret_ok] * w_ret) else NA_real_
  sector_ret_ew <- safe_mean(LogReturn)
  
  p_var <- if (any(cap_ok)) safe_weighted_mean(ParkinsonVar[cap_ok], LagMarketCapKES[cap_ok]) else NA_real_
  gk_var <- if (any(cap_ok)) safe_weighted_mean(GarmanKlassVar[cap_ok], LagMarketCapKES[cap_ok]) else NA_real_
  rs_var <- if (any(cap_ok)) safe_weighted_mean(RogersSatchellVar[cap_ok], LagMarketCapKES[cap_ok]) else NA_real_
  
  hhi <- if (length(w_cap) > 0L) sum(w_cap^2) else NA_real_
  
  list(
    SectorReturn_MCW = sector_ret_mcw,
    SectorReturn_EW = sector_ret_ew,
    ParkinsonVol = if (is.finite(p_var) && p_var >= 0) sqrt(p_var) else NA_real_,
    GarmanKlassVol = if (is.finite(gk_var) && gk_var >= 0) sqrt(gk_var) else NA_real_,
    RogersSatchellVol = if (is.finite(rs_var) && rs_var >= 0) sqrt(rs_var) else NA_real_,
    Turnover_MCW = safe_weighted_mean(Turnover, LagMarketCapKES),
    AmihudILLIQ_Median = if (any(is.finite(AmihudILLIQ))) median(AmihudILLIQ, na.rm = TRUE) else NA_real_,
    BreadthNegative = if (any(is.finite(LogReturn))) mean(LogReturn < 0, na.rm = TRUE) else NA_real_,
    ZeroReturnShare = if (any(!is.na(ZeroReturn))) mean(ZeroReturn, na.rm = TRUE) else NA_real_,
    ZeroVolumeShare = if (any(!is.na(ZeroVolume))) mean(ZeroVolume, na.rm = TRUE) else NA_real_,
    ReturnDispersion = safe_sd(LogReturn),
    MarketCapKES = sum(MarketCapKES[is.finite(MarketCapKES)], na.rm = TRUE),
    HHI = hhi,
    NStocksObserved = uniqueN(Stock),
    NStocksReturn = sum(is.finite(LogReturn)),
    NCorporateActions = sum(CorporateActionFlag, na.rm = TRUE)
  )
}, by = .(Sector, Date)]

setorder(sector_daily, Sector, Date)

# Complete sector-date grid WITHOUT imputing returns.
all_sector_grid <- CJ(
  Sector = sort(unique(clean$Sector)),
  Date = calendar$Date,
  unique = TRUE
)

sector_panel <- merge(
  all_sector_grid,
  sector_daily,
  by = c("Sector", "Date"),
  all.x = TRUE,
  sort = FALSE
)

setorder(sector_panel, Sector, Date)
sector_panel[, SectorObserved := as.integer(!is.na(NStocksObserved))]
sector_panel[, SectorReturnAvailable := as.integer(is.finite(SectorReturn_MCW))]

# Constituent-coverage safeguard.
# A valid return from one isolated stock is NOT sufficient to call the result a
# sector return for crash modelling. The raw sector return is retained for
# descriptive purposes, while SectorReturn_Model is used by GARCH/EVT.
sector_panel[, ConstituentReturnShare := fifelse(
  is.finite(NStocksObserved) & NStocksObserved > 0,
  NStocksReturn / NStocksObserved,
  NA_real_
)]

sector_panel[, SectorDayEligible := as.integer(
  SectorReturnAvailable == 1L &
    NStocksReturn >= PRIMARY_MIN_DAILY_RETURNS &
    is.finite(ConstituentReturnShare) &
    ConstituentReturnShare >= PRIMARY_MIN_DAILY_RETURN_SHARE
)]

sector_panel[, SectorReturn_Model := fifelse(
  SectorDayEligible == 1L,
  SectorReturn_MCW,
  NA_real_
)]

# ==============================================================================
# 9. MARKET-WIDE INTERNAL FEATURES (DERIVED FROM THE SAME MASTER FILE)
# ==============================================================================

market_daily <- clean[, {
  ret_ok <- is.finite(LogReturn) & is.finite(LagMarketCapKES) & LagMarketCapKES > 0
  
  list(
    MarketReturn_MCW = if (any(ret_ok)) {
      sum(LogReturn[ret_ok] * LagMarketCapKES[ret_ok]) / sum(LagMarketCapKES[ret_ok])
    } else NA_real_,
    MarketBreadthNegative = if (any(is.finite(LogReturn))) mean(LogReturn < 0, na.rm = TRUE) else NA_real_,
    MarketZeroReturnShare = if (any(!is.na(ZeroReturn))) mean(ZeroReturn, na.rm = TRUE) else NA_real_,
    MarketZeroVolumeShare = if (any(!is.na(ZeroVolume))) mean(ZeroVolume, na.rm = TRUE) else NA_real_,
    MarketReturnDispersion = safe_sd(LogReturn),
    MarketTurnover_MCW = safe_weighted_mean(Turnover, LagMarketCapKES),
    MarketAmihudILLIQ_Median = if (any(is.finite(AmihudILLIQ))) median(AmihudILLIQ, na.rm = TRUE) else NA_real_,
    MarketCapKES_Total = sum(MarketCapKES[is.finite(MarketCapKES)], na.rm = TRUE),
    NStocksObservedMarket = uniqueN(Stock),
    NStocksReturnMarket = sum(is.finite(LogReturn))
  )
}, by = Date]

cross_sector_daily <- sector_panel[, .(
  NSectorsObserved = sum(SectorObserved == 1L, na.rm = TRUE),
  NSectorsWithReturn = sum(SectorReturnAvailable == 1L, na.rm = TRUE),
  NSectorsDown = sum(SectorReturn_MCW < 0, na.rm = TRUE),
  CrossSectorReturnDispersion = safe_sd(SectorReturn_MCW)
), by = Date]

market_daily <- merge(market_daily, cross_sector_daily, by = "Date", all = TRUE)

sector_panel <- merge(sector_panel, market_daily, by = "Date", all.x = TRUE, sort = FALSE)
setorder(sector_panel, Sector, Date)

# ==============================================================================
# 10. SECTOR COVERAGE AND PRIMARY-SECTOR ELIGIBILITY
# ==============================================================================

n_market_dates <- nrow(calendar)

sector_composition <- clean[, .(
  NStocks = uniqueN(Stock),
  FirstDate = min(Date),
  LastDate = max(Date),
  NObservedRows = .N
), by = Sector]

sector_coverage <- sector_panel[, .(
  MarketDates = n_market_dates,
  SectorDatesObserved = sum(SectorObserved == 1L, na.rm = TRUE),
  ReturnDates = sum(SectorReturnAvailable == 1L, na.rm = TRUE),
  EligibleModelDays = sum(SectorDayEligible == 1L, na.rm = TRUE),
  DateCoverage = mean(SectorObserved == 1L, na.rm = TRUE),
  ReturnCoverage = mean(SectorReturnAvailable == 1L, na.rm = TRUE),
  EligibleDayCoverage = mean(SectorDayEligible == 1L, na.rm = TRUE),
  # Force integer-count medians to double. data.table requires each grouped
  # result column to have an identical storage type across sectors; median()
  # can return integer for an odd-sized group and double for an even-sized group.
  MedianNStocksObserved = as.numeric(median(NStocksObserved, na.rm = TRUE)),
  MedianNStocksReturn = as.numeric(median(NStocksReturn, na.rm = TRUE)),
  MedianConstituentReturnShare = as.numeric(median(ConstituentReturnShare, na.rm = TRUE))
), by = Sector]

sector_coverage <- merge(
  sector_coverage,
  sector_composition[, .(Sector, NStocks, FirstDate, LastDate)],
  by = "Sector",
  all.x = TRUE
)

sector_coverage[, PrimarySector := (
  NStocks >= PRIMARY_MIN_STOCKS &
    EligibleDayCoverage >= PRIMARY_MIN_ELIGIBLE_DAY_COVERAGE
)]

sector_coverage[, ExclusionReason := fifelse(
  PrimarySector == TRUE,
  "Primary",
  fifelse(
    NStocks < PRIMARY_MIN_STOCKS,
    paste0("Fewer than ", PRIMARY_MIN_STOCKS, " constituent stocks"),
    paste0(
      "Eligible sector-day coverage below ",
      sprintf("%.0f%%", 100 * PRIMARY_MIN_ELIGIBLE_DAY_COVERAGE)
    )
  )
)]

sector_panel <- merge(
  sector_panel,
  sector_coverage[, .(
    Sector, NStocks, ReturnCoverage, EligibleDayCoverage,
    PrimarySector, ExclusionReason
  )],
  by = "Sector",
  all.x = TRUE,
  sort = FALSE
)

setorder(sector_panel, Sector, Date)

log_msg(
  "Primary sectors under pre-specified criteria:",
  paste(sector_coverage[PrimarySector == TRUE, Sector], collapse = " | ")
)

# ==============================================================================
# 11. DESCRIPTIVE TABLES
# ==============================================================================

sector_summary <- sector_panel[is.finite(SectorReturn_MCW), .(
  NReturnDays = .N,
  AnnualizedMean = mean(SectorReturn_MCW) * TRADING_DAYS_PER_YEAR,
  AnnualizedSD = sd(SectorReturn_MCW) * sqrt(TRADING_DAYS_PER_YEAR),
  Skewness = safe_skew(SectorReturn_MCW),
  Kurtosis = safe_kurt(SectorReturn_MCW),
  MinimumReturn = safe_min(SectorReturn_MCW),
  MaximumReturn = safe_max(SectorReturn_MCW),
  MeanParkinsonVol = safe_mean(ParkinsonVol),
  MeanTurnover = safe_mean(Turnover_MCW),
  MedianAmihudILLIQ = if (any(is.finite(AmihudILLIQ_Median))) median(AmihudILLIQ_Median, na.rm = TRUE) else NA_real_,
  MeanBreadthNegative = safe_mean(BreadthNegative),
  MeanZeroReturnShare = safe_mean(ZeroReturnShare),
  MeanZeroVolumeShare = safe_mean(ZeroVolumeShare),
  MeanReturnDispersion = safe_mean(ReturnDispersion),
  MeanHHI = safe_mean(HHI)
), by = .(Sector, PrimarySector)]

corporate_action_table <- clean[CorporateActionFlag == TRUE, .(
  Date,
  Sector,
  Stock,
  BonusIssueShares,
  IssuedShares,
  LagIssuedShares,
  ShareRatio,
  BonusFlag,
  ShareChangeFlag,
  RawLogReturn
)]

stock_thin_trading <- clean[, .(
  Sector = first(Sector),
  FirstDate = min(Date),
  LastDate = max(Date),
  NRows = .N,
  NDailyReturns = sum(is.finite(LogReturn)),
  ZeroReturnPct = 100 * mean(ZeroReturn, na.rm = TRUE),
  ZeroVolumePct = 100 * mean(ZeroVolume, na.rm = TRUE),
  MedianVolume = if (any(is.finite(Volume))) median(Volume, na.rm = TRUE) else NA_real_
), by = Stock]

# Proposed temporal split marker. EVT labels are joined later; rows before the
# EVT history is sufficient will naturally have missing Crash_Main.
sector_panel[, Split := fifelse(
  Date <= TRAIN_END, "Train",
  fifelse(Date <= VALID_END, "Validation", "Test")
)]

# Write audit/descriptive CSV tables.
write_table_csv(audit_overview, "Table_R01_raw_audit_overview.csv")
write_table_csv(missingness_raw, "Table_R02_raw_missingness.csv")
write_table_csv(sector_composition, "Table_R03_sector_composition.csv")
write_table_csv(sector_coverage, "Table_R04_sector_coverage_primary_flag.csv")
write_table_csv(sector_summary, "Table_R05_sector_descriptive_statistics.csv")
write_table_csv(stock_thin_trading, "Table_R06_stock_thin_trading_diagnostics.csv")
write_table_csv(corporate_action_table, "Table_R07_corporate_action_flags.csv")

if (nrow(critical_rows) > 0L) {
  write_table_csv(critical_rows, "Audit_excluded_critical_missing_rows.csv")
}
if (nrow(invalid_ohlc_rows) > 0L) {
  write_table_csv(invalid_ohlc_rows, "Audit_invalid_OHLC_rows_retained_for_close_return.csv")
}

# ==============================================================================
# 12. EXPLORATORY / SUPPORTING FIGURES FROM R
# ==============================================================================

# Figure R01: annualized sector volatility.
fig_vol <- ggplot(
  sector_summary,
  aes(x = reorder(Sector, AnnualizedSD), y = AnnualizedSD, shape = PrimarySector)
) +
  geom_point(size = 3) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(
    x = NULL,
    y = "Annualized standard deviation",
    shape = "Primary sector",
    title = "Sectoral return volatility"
  ) +
  theme_minimal(base_size = 11)

save_plot_both(fig_vol, "Figure_R01_sector_annualized_volatility", width = 10, height = 6.5)

# Figure R02: sector zero-return share.
fig_zero <- ggplot(
  sector_summary,
  aes(x = reorder(Sector, MeanZeroReturnShare), y = MeanZeroReturnShare, shape = PrimarySector)
) +
  geom_point(size = 3) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(
    x = NULL,
    y = "Mean daily zero-return share",
    shape = "Primary sector",
    title = "Thin-trading indicator by sector"
  ) +
  theme_minimal(base_size = 11)

save_plot_both(fig_zero, "Figure_R02_sector_zero_return_share", width = 10, height = 6.5)

# Figure R03: daily sector returns for primary sectors.
fig_ret <- ggplot(
  sector_panel[PrimarySector == TRUE & is.finite(SectorReturn_Model)],
  aes(x = Date, y = SectorReturn_Model)
) +
  geom_line(linewidth = 0.25) +
  facet_wrap(~Sector, scales = "free_y", ncol = 2) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    x = NULL,
    y = "Eligible market-cap-weighted log return",
    title = "Primary-sector daily returns after constituent-coverage screening"
  ) +
  theme_minimal(base_size = 10)

save_plot_both(fig_ret, "Figure_R03_primary_sector_returns", width = 12, height = 11)

# ==============================================================================
# 13. GARCH CANDIDATE COMPARISON ON PRE-SPECIFIED PRE-2022 TRAINING DATA
# ==============================================================================

garch_candidate_results <- data.table()

if (RUN_GARCH_EVT) {
  log_msg("Starting GARCH candidate comparison.")
  
  candidate_rows <- list()
  
  for (s in sort(unique(sector_panel[PrimarySector == TRUE, Sector]))) {
    x <- sector_panel[
      Sector == s &
        PrimarySector == TRUE &
        Date <= TRAIN_END &
        is.finite(SectorReturn_Model),
      SectorReturn_Model
    ] * 100
    
    if (length(x) < GARCH_MIN_HISTORY) {
      next
    }
    
    for (m in GARCH_CANDIDATES) {
      candidate_rows[[length(candidate_rows) + 1L]] <-
        fit_garch_candidate(
          x_pct = x,
          sector_name = s,
          model_name = m,
          dist_name = GARCH_CANDIDATE_DIST
        )
    }
  }
  
  if (length(candidate_rows) > 0L) {
    garch_candidate_results <- rbindlist(candidate_rows, use.names = TRUE, fill = TRUE)
    garch_candidate_results[, BestBICWithinSector := BIC == min(BIC, na.rm = TRUE), by = Sector]
    write_table_csv(garch_candidate_results, "Table_R08_GARCH_candidate_comparison.csv")
  }
}

# ==============================================================================
# 14. ROLLING GJR-GARCH + ROLLING EVT CRASH IDENTIFICATION
# ==============================================================================

garch_evt_panel <- data.table()
evt_parameter_history <- data.table()
garch_evt_errors <- data.table()

if (RUN_GARCH_EVT) {
  log_msg("Starting rolling", GARCH_MAIN_MODEL, "+ EVT across sectors.")
  
  panel_rows <- list()
  param_rows <- list()
  error_rows <- list()
  
  for (s in sort(unique(sector_panel[PrimarySector == TRUE, Sector]))) {
    res <- run_sector_garch_evt(s, sector_panel)
    
    if (nrow(res$panel) > 0L) {
      panel_rows[[length(panel_rows) + 1L]] <- res$panel
    }
    if (nrow(res$evt_params) > 0L) {
      param_rows[[length(param_rows) + 1L]] <- res$evt_params
    }
    if (nrow(res$error) > 0L) {
      error_rows[[length(error_rows) + 1L]] <- res$error
    }
  }
  
  if (length(panel_rows) > 0L) {
    garch_evt_panel <- rbindlist(panel_rows, use.names = TRUE, fill = TRUE)
  }
  if (length(param_rows) > 0L) {
    evt_parameter_history <- rbindlist(param_rows, use.names = TRUE, fill = TRUE)
  }
  if (length(error_rows) > 0L) {
    garch_evt_errors <- rbindlist(error_rows, use.names = TRUE, fill = TRUE)
  }
}

# ------------------------------------------------------------------------------
# 14.1 EVT ESTIMATION INTEGRITY GUARD
# ------------------------------------------------------------------------------
# This guard is deliberately strict. It would have caught the Version 1.0
# problem in which every rolling GPD shape estimate remained exactly at 0.10.
# No crash labels are exported if rolling EVT parameters are effectively frozen.
evt_integrity <- data.table()

if (nrow(evt_parameter_history) > 0L) {
  evt_integrity <- evt_parameter_history[, .(
    NRefits = .N,
    UniqueShape_6dp = uniqueN(round(GPD_Shape, 6)),
    ShapeSD = stats::sd(GPD_Shape, na.rm = TRUE),
    MinShape = min(GPD_Shape, na.rm = TRUE),
    MaxShape = max(GPD_Shape, na.rm = TRUE),
    UniqueScale_6dp = uniqueN(round(GPD_Scale, 6)),
    ScaleSD = stats::sd(GPD_Scale, na.rm = TRUE),
    BoundaryFits = sum(AtBoundary %in% TRUE, na.rm = TRUE),
    NonConverged = sum(Convergence != 0L, na.rm = TRUE),
    MeanNLLImprovement = mean(NLL_Improvement, na.rm = TRUE)
  ), by = Sector]
  
  write_table_csv(
    evt_integrity,
    "Table_R08B_EVT_estimation_integrity_guard.csv"
  )
  
  bad_evt <- evt_integrity[
    NRefits >= 5L &
      (
        UniqueShape_6dp <= 1L |
          !is.finite(ShapeSD) |
          ShapeSD < 1e-8 |
          UniqueScale_6dp <= 1L |
          !is.finite(ScaleSD) |
          ScaleSD < 1e-8
      )
  ]
  
  if (nrow(bad_evt) > 0L) {
    stop(
      paste0(
        "EVT INTEGRITY FAILURE: rolling GPD parameters are effectively frozen ",
        "for: ", paste(bad_evt$Sector, collapse = " | "),
        ". Crash labels have NOT been frozen/exported. Review ",
        "Table_R08B_EVT_estimation_integrity_guard.csv."
      )
    )
  }
  
  if (any(evt_integrity$BoundaryFits > 0L, na.rm = TRUE)) {
    log_msg(
      "WARNING: Some GPD shape estimates touched configured bounds. ",
      "Review EVT integrity/parameter-history tables."
    )
  }
} else if (RUN_GARCH_EVT) {
  stop("EVT INTEGRITY FAILURE: no valid rolling EVT parameter estimates were produced.")
}

# ==============================================================================
# 15. JOIN ECONOMETRIC OUTPUTS TO THE MASTER SECTOR PANEL
# ==============================================================================

if (nrow(garch_evt_panel) > 0L) {
  final_panel <- merge(
    sector_panel,
    garch_evt_panel,
    by = c("Date", "Sector"),
    all.x = TRUE,
    sort = FALSE
  )
} else {
  final_panel <- copy(sector_panel)
}

setorder(final_panel, Sector, Date)

# Defensive primary-label aliases.
# Version 2 exported CrashThreshold_0025 but did not always carry the convenience
# alias CrashThreshold_Main into final_panel. The reviewer-revision diagnostics
# use CrashThreshold_Main, so reconstruct the alias deterministically when needed.
if (!"CrashThreshold_Main" %in% names(final_panel) &&
    "CrashThreshold_0025" %in% names(final_panel)) {
  final_panel[, CrashThreshold_Main := CrashThreshold_0025]
}
if (!"Crash_Main" %in% names(final_panel) &&
    "Crash_0025" %in% names(final_panel)) {
  final_panel[, Crash_Main := Crash_0025]
}

# Event feasibility by sector and temporal split.
if ("Crash_Main" %in% names(final_panel)) {
  crash_counts <- final_panel[!is.na(Crash_Main), .(
    LabelledDays = .N,
    CrashEvents = sum(Crash_Main == 1L, na.rm = TRUE),
    CrashRate = mean(Crash_Main == 1L, na.rm = TRUE)
  ), by = .(Sector, PrimarySector, Split)]
  
  crash_counts_wide <- dcast(
    crash_counts,
    Sector + PrimarySector ~ Split,
    value.var = c("LabelledDays", "CrashEvents", "CrashRate"),
    fill = NA
  )
  
  crash_counts_tail <- final_panel[, {
    out <- list()
    if ("Crash_001" %in% names(final_panel)) {
      out$Crash_1pct <- sum(Crash_001 == 1L, na.rm = TRUE)
      out$Labelled_1pct <- sum(!is.na(Crash_001))
    }
    if ("Crash_0025" %in% names(final_panel)) {
      out$Crash_2_5pct <- sum(Crash_0025 == 1L, na.rm = TRUE)
      out$Labelled_2_5pct <- sum(!is.na(Crash_0025))
    }
    if ("Crash_005" %in% names(final_panel)) {
      out$Crash_5pct <- sum(Crash_005 == 1L, na.rm = TRUE)
      out$Labelled_5pct <- sum(!is.na(Crash_005))
    }
    out
  }, by = .(Sector, PrimarySector)]
  
  write_table_csv(crash_counts, "Table_R09_crash_event_counts_by_split.csv")
  write_table_csv(crash_counts_wide, "Table_R10_crash_event_counts_by_sector_wide.csv")
  write_table_csv(crash_counts_tail, "Table_R11_crash_counts_alternative_EVT_tails.csv")
} else {
  crash_counts <- data.table()
  crash_counts_wide <- data.table()
  crash_counts_tail <- data.table()
}

if (nrow(evt_parameter_history) > 0L) {
  evt_summary <- evt_parameter_history[, .(
    NRefits = .N,
    MeanThreshold_u = mean(Threshold_u, na.rm = TRUE),
    MeanExceedN = mean(ExceedN, na.rm = TRUE),
    MeanExceedRate = mean(ExceedRate, na.rm = TRUE),
    MeanScale = mean(GPD_Scale, na.rm = TRUE),
    SDScale = stats::sd(GPD_Scale, na.rm = TRUE),
    MeanShape = mean(GPD_Shape, na.rm = TRUE),
    SDShape = stats::sd(GPD_Shape, na.rm = TRUE),
    MinShape = min(GPD_Shape, na.rm = TRUE),
    MaxShape = max(GPD_Shape, na.rm = TRUE),
    UniqueShape_6dp = uniqueN(round(GPD_Shape, 6)),
    BoundaryFits = sum(AtBoundary %in% TRUE, na.rm = TRUE),
    MeanNLLImprovement = mean(NLL_Improvement, na.rm = TRUE),
    NonConverged = sum(Convergence != 0L, na.rm = TRUE)
  ), by = Sector]
  
  write_table_csv(evt_summary, "Table_R12_EVT_parameter_summary.csv")
  write_table_csv(evt_parameter_history, "Table_R13_EVT_parameter_refit_history.csv")
}

if (nrow(garch_evt_errors) > 0L) {
  write_table_csv(garch_evt_errors, "Table_R14_GARCH_EVT_errors.csv")
}

# ==============================================================================
# 16. GARCH/EVT SUPPORTING FIGURES
# ==============================================================================

if (
  "GARCH_Sigma" %in% names(final_panel) &&
  any(is.finite(final_panel$GARCH_Sigma))
) {
  fig_garch <- ggplot(
    final_panel[PrimarySector == TRUE & is.finite(GARCH_Sigma)],
    aes(x = Date, y = GARCH_Sigma)
  ) +
    geom_line(linewidth = 0.3) +
    facet_wrap(~Sector, scales = "free_y", ncol = 2) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    labs(
      x = NULL,
      y = "Conditional daily volatility",
      title = "Rolling GJR-GARCH conditional volatility"
    ) +
    theme_minimal(base_size = 10)
  
  save_plot_both(fig_garch, "Figure_R04_rolling_GJR_GARCH_volatility", width = 12, height = 11)
}

if (
  all(c("CrashThreshold_Main", "Crash_Main") %in% names(final_panel)) &&
  any(is.finite(final_panel$CrashThreshold_Main))
) {
  fig_evt <- ggplot(
    final_panel[
      PrimarySector == TRUE &
        is.finite(SectorReturn_Model) &
        is.finite(CrashThreshold_Main)
    ],
    aes(x = Date)
  ) +
    geom_line(aes(y = SectorReturn_Model), linewidth = 0.25) +
    geom_line(aes(y = CrashThreshold_Main), linewidth = 0.35, linetype = "dashed") +
    geom_point(
      data = final_panel[
        PrimarySector == TRUE &
          Crash_Main == 1L &
          is.finite(SectorReturn_Model)
      ],
      aes(y = SectorReturn_Model),
      size = 1.2
    ) +
    facet_wrap(~Sector, scales = "free_y", ncol = 2) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      x = NULL,
      y = "Daily return / dynamic EVT threshold",
      title = "Volatility-adjusted EVT crash identification"
    ) +
    theme_minimal(base_size = 10)
  
  save_plot_both(fig_evt, "Figure_R05_dynamic_EVT_crash_thresholds", width = 12, height = 11)
}


# ==============================================================================
# 16.1 REVIEWER-REVISION DIAGNOSTICS AND ROBUSTNESS
# ==============================================================================
# IMPORTANT: legacy variables named Crash_* are retained ONLY for downstream
# code compatibility. In the revised manuscript they are described as
# conditional sector tail-event indicators, not conventional market crashes.

if (DATA_SOURCE_NOTE == "TO_BE_CONFIRMED") {
  log_msg(
    "WARNING: NSE_DATA_SOURCE_NOTE is not set. Record the exact data source ",
    "before the final archival run."
  )
}
if (DATA_LICENSE_NOTE == "TO_BE_CONFIRMED") {
  log_msg(
    "WARNING: NSE_DATA_LICENSE_NOTE is not set. Record the licence/usage basis ",
    "before the final archival run."
  )
}

# ------------------------------------------------------------------------------
# 16.1.1 Revised chronological roles — preserve legacy Split for compatibility
# ------------------------------------------------------------------------------
reviewer_required_cols <- c(
  "Date", "Sector", "PrimarySector", "Split",
  "SectorReturn_Model", "Crash_Main", "CrashThreshold_Main",
  "GARCH_Sigma", "StdInnovation"
)
reviewer_missing_cols <- setdiff(reviewer_required_cols, names(final_panel))
if (length(reviewer_missing_cols) > 0L) {
  stop(
    paste0(
      "Reviewer diagnostics cannot start because final_panel is missing: ",
      paste(reviewer_missing_cols, collapse = ", ")
    )
  )
}

final_panel[, RevisionPhase := fifelse(
  Date < INNER_TUNE_START, "TrainFit",
  fifelse(
    Date <= TRAIN_END, "TrainTune",
    fifelse(Date <= VALID_END, "Calibration", "Test")
  )
)]

revision_phase_counts <- final_panel[PrimarySector == TRUE, .(
  CalendarRows = .N,
  LabelledDays = sum(!is.na(Crash_Main)),
  OneDayEvents = sum(Crash_Main == 1L, na.rm = TRUE),
  OneDayEventRate = if (sum(!is.na(Crash_Main)) > 0L) {
    mean(Crash_Main == 1L, na.rm = TRUE)
  } else NA_real_
), by = RevisionPhase]

write_table_csv(
  revision_phase_counts,
  "Table_R16_reviewer_revision_phase_counts.csv"
)

# ------------------------------------------------------------------------------
# 16.1.2 Historical observed constituent universe / survivorship diagnostics
# ------------------------------------------------------------------------------
# These tables describe the constituent universe PRESENT IN THE UPLOADED MASTER
# FILE. They do not reconstruct unavailable delisted/suspended histories.
observed_constituent_history <- clean[, .(
  FirstObservedDate = min(Date, na.rm = TRUE),
  LastObservedDate = max(Date, na.rm = TRUE),
  NObservedRows = .N,
  NReturnRows = sum(is.finite(LogReturn)),
  ZeroReturnPct = 100 * mean(ZeroReturn, na.rm = TRUE),
  ZeroVolumePct = 100 * mean(ZeroVolume, na.rm = TRUE)
), by = .(Stock, Sector)]

sector_composition_yearly <- clean[, .(
  NStocksObserved = uniqueN(Stock),
  ConstituentsObserved = paste(sort(unique(Stock)), collapse = " | ")
), by = .(Year = lubridate::year(Date), Sector)]

stock_sector_classification_audit <- clean[, .(
  NSectorsObserved = uniqueN(Sector),
  SectorsObserved = paste(sort(unique(Sector)), collapse = " | "),
  FirstObservedDate = min(Date, na.rm = TRUE),
  LastObservedDate = max(Date, na.rm = TRUE)
), by = Stock]

write_table_csv(
  observed_constituent_history,
  "Table_R17_observed_constituent_history.csv"
)
write_table_csv(
  sector_composition_yearly,
  "Table_R18_sector_composition_by_year.csv"
)
write_table_csv(
  stock_sector_classification_audit,
  "Table_R19_stock_sector_classification_audit.csv"
)

# ------------------------------------------------------------------------------
# 16.1.3 Economic magnitude of primary tail events
# ------------------------------------------------------------------------------
tail_event_rows <- final_panel[
  PrimarySector == TRUE &
    Crash_Main == 1L &
    is.finite(SectorReturn_Model) &
    is.finite(CrashThreshold_Main),
  .(
    Date,
    Sector,
    Split,
    RevisionPhase,
    SectorReturn = SectorReturn_Model,
    DynamicThreshold = CrashThreshold_Main,
    ThresholdShortfall = CrashThreshold_Main - SectorReturn_Model,
    AbsoluteReturn = abs(SectorReturn_Model),
    GARCH_Sigma,
    StdInnovation
  )
]

tail_event_magnitude_summary <- if (nrow(tail_event_rows) > 0L) {
  tail_event_rows[, .(
    NEvents = .N,
    MeanReturn = mean(SectorReturn, na.rm = TRUE),
    MedianReturn = median(SectorReturn, na.rm = TRUE),
    Q25Return = as.numeric(quantile(SectorReturn, 0.25, na.rm = TRUE, names = FALSE)),
    Q05Return = as.numeric(quantile(SectorReturn, 0.05, na.rm = TRUE, names = FALSE)),
    WorstReturn = min(SectorReturn, na.rm = TRUE),
    MeanDynamicThreshold = mean(DynamicThreshold, na.rm = TRUE),
    MeanThresholdShortfall = mean(ThresholdShortfall, na.rm = TRUE),
    MedianThresholdShortfall = median(ThresholdShortfall, na.rm = TRUE)
  ), by = .(Sector, RevisionPhase)]
} else {
  data.table(
    Sector = character(),
    RevisionPhase = character(),
    NEvents = integer(),
    MeanReturn = numeric(),
    MedianReturn = numeric(),
    Q25Return = numeric(),
    Q05Return = numeric(),
    WorstReturn = numeric(),
    MeanDynamicThreshold = numeric(),
    MeanThresholdShortfall = numeric(),
    MedianThresholdShortfall = numeric()
  )
}

write_table_csv(
  tail_event_rows,
  "Table_R20_primary_tail_event_observations.csv"
)
write_table_csv(
  tail_event_magnitude_summary,
  "Table_R21_primary_tail_event_magnitude_summary.csv"
)

# ------------------------------------------------------------------------------
# 16.1.4 EVT threshold stability, mean residual life, PIT/tail-fit diagnostics,
#        parameter uncertainty, and fallback-frequency reconstruction
# ------------------------------------------------------------------------------
gpd_cdf_eval <- function(y, scale, shape) {
  y <- as.numeric(y)
  out <- rep(NA_real_, length(y))
  ok <- is.finite(y) & y >= 0 & is.finite(scale) & scale > 0 & is.finite(shape)
  if (!any(ok)) return(out)
  yy <- y[ok]
  if (abs(shape) < 1e-8) {
    out[ok] <- 1 - exp(-yy / scale)
  } else {
    support <- 1 + shape * yy / scale
    good <- support > 0
    tmp <- rep(NA_real_, length(yy))
    tmp[good] <- 1 - support[good]^(-1 / shape)
    out[ok] <- tmp
  }
  pmin(pmax(out, 0), 1)
}

gpd_quantile_eval <- function(p, scale, shape) {
  p <- as.numeric(p)
  out <- rep(NA_real_, length(p))
  ok <- is.finite(p) & p > 0 & p < 1 & is.finite(scale) & scale > 0 & is.finite(shape)
  if (!any(ok)) return(out)
  pp <- p[ok]
  if (abs(shape) < 1e-8) {
    out[ok] <- -scale * log(1 - pp)
  } else {
    out[ok] <- (scale / shape) * ((1 - pp)^(-shape) - 1)
  }
  out
}

gpd_parameter_uncertainty <- function(excess, scale, shape, conf_level = EVT_CONF_LEVEL) {
  y <- excess[is.finite(excess) & excess > 0]
  if (length(y) < EVT_MIN_EXCEEDANCES || !is.finite(scale) || scale <= 0 || !is.finite(shape)) {
    return(data.table(
      ScaleSE = NA_real_, ShapeSE = NA_real_,
      ScaleCI_L = NA_real_, ScaleCI_U = NA_real_,
      ShapeCI_L = NA_real_, ShapeCI_U = NA_real_,
      HessianOK = FALSE
    ))
  }
  h <- tryCatch(
    stats::optimHess(
      par = c(log(scale), shape),
      fn = gpd_negloglik,
      y = y
    ),
    error = function(e) NULL
  )
  if (is.null(h) || any(!is.finite(h))) {
    return(data.table(
      ScaleSE = NA_real_, ShapeSE = NA_real_,
      ScaleCI_L = NA_real_, ScaleCI_U = NA_real_,
      ShapeCI_L = NA_real_, ShapeCI_U = NA_real_,
      HessianOK = FALSE
    ))
  }
  vc <- tryCatch(solve(h), error = function(e) NULL)
  if (is.null(vc) || any(!is.finite(vc)) || any(diag(vc) <= 0)) {
    return(data.table(
      ScaleSE = NA_real_, ShapeSE = NA_real_,
      ScaleCI_L = NA_real_, ScaleCI_U = NA_real_,
      ShapeCI_L = NA_real_, ShapeCI_U = NA_real_,
      HessianOK = FALSE
    ))
  }
  zc <- stats::qnorm((1 + conf_level) / 2)
  se_log_scale <- sqrt(vc[1, 1])
  se_shape <- sqrt(vc[2, 2])
  data.table(
    ScaleSE = scale * se_log_scale,
    ShapeSE = se_shape,
    ScaleCI_L = scale * exp(-zc * se_log_scale),
    ScaleCI_U = scale * exp(zc * se_log_scale),
    ShapeCI_L = shape - zc * se_shape,
    ShapeCI_U = shape + zc * se_shape,
    HessianOK = TRUE
  )
}

evt_threshold_stability <- data.table()
evt_pit_diagnostics <- data.table()
evt_parameter_uncertainty <- data.table()
evt_refit_attempts <- data.table()
evt_refit_summary <- data.table()

if (RUN_REVIEWER_DIAGNOSTICS && nrow(evt_parameter_history) > 0L && nrow(garch_evt_panel) > 0L) {
  diag_refits <- evt_parameter_history[, {
    idx <- sort(unique(c(
      1L,
      seq.int(1L, .N, by = EVT_DIAGNOSTIC_REFIT_STRIDE),
      .N
    )))
    .SD[idx]
  }, by = Sector]
  
  stab_rows <- list()
  pit_rows <- list()
  unc_rows <- list()
  
  for (rr in seq_len(nrow(diag_refits))) {
    rp <- diag_refits[rr]
    loss_hist <- -garch_evt_panel[
      Sector == rp$Sector &
        Date < rp$RefitDate &
        is.finite(StdInnovation),
      StdInnovation
    ]
    loss_hist <- loss_hist[is.finite(loss_hist)]
    
    # Primary-fit uncertainty and PIT diagnostics.
    excess_primary <- loss_hist[loss_hist > rp$Threshold_u] - rp$Threshold_u
    unc <- gpd_parameter_uncertainty(
      excess_primary,
      rp$GPD_Scale,
      rp$GPD_Shape,
      EVT_CONF_LEVEL
    )
    unc_rows[[length(unc_rows) + 1L]] <- cbind(
      rp[, .(Sector, RefitID, RefitDate, HistN, ExceedN, GPD_Scale, GPD_Shape)],
      unc
    )
    
    pit <- gpd_cdf_eval(excess_primary, rp$GPD_Scale, rp$GPD_Shape)
    pit <- pit[is.finite(pit)]
    ks <- if (length(pit) >= 10L) {
      tryCatch(
        suppressWarnings(stats::ks.test(pit, "punif")),
        error = function(e) NULL
      )
    } else NULL
    theo <- if (length(excess_primary) >= 3L) {
      gpd_quantile_eval(stats::ppoints(length(excess_primary)), rp$GPD_Scale, rp$GPD_Shape)
    } else numeric(0)
    qq_cor <- if (
      length(theo) == length(excess_primary) &&
      length(theo) >= 3L &&
      all(is.finite(theo))
    ) {
      suppressWarnings(stats::cor(sort(excess_primary), theo))
    } else NA_real_
    
    pit_rows[[length(pit_rows) + 1L]] <- data.table(
      Sector = rp$Sector,
      RefitID = rp$RefitID,
      RefitDate = rp$RefitDate,
      NExceed = length(excess_primary),
      PIT_Mean = if (length(pit)) mean(pit) else NA_real_,
      PIT_Variance = if (length(pit) > 1L) stats::var(pit) else NA_real_,
      PIT_KS_D = if (!is.null(ks)) as.numeric(ks$statistic) else NA_real_,
      PIT_KS_p = if (!is.null(ks)) as.numeric(ks$p.value) else NA_real_,
      TailQQCorrelation = qq_cor
    )
    
    # Threshold-stability / MRL grid.
    for (q in EVT_DIAGNOSTIC_THRESHOLD_PROBS) {
      u_q <- as.numeric(stats::quantile(
        loss_hist,
        probs = q,
        na.rm = TRUE,
        names = FALSE,
        type = 8
      ))
      excess_q <- loss_hist[loss_hist > u_q] - u_q
      fit_q <- if (length(excess_q) >= EVT_MIN_EXCEEDANCES) {
        fit_gpd_mle(excess_q)
      } else {
        list(
          scale = NA_real_, shape = NA_real_, convergence = 99L,
          n_exceed = length(excess_q), loglik = NA_real_,
          method = "insufficient_data", nll_improvement = NA_real_,
          at_boundary = NA
        )
      }
      stab_rows[[length(stab_rows) + 1L]] <- data.table(
        Sector = rp$Sector,
        RefitID = rp$RefitID,
        RefitDate = rp$RefitDate,
        HistN = length(loss_hist),
        ThresholdProb = q,
        Threshold_u = u_q,
        ExceedN = length(excess_q),
        ExceedRate = length(excess_q) / length(loss_hist),
        MeanResidualLife = if (length(excess_q)) mean(excess_q) else NA_real_,
        GPD_Scale = fit_q$scale,
        GPD_Shape = fit_q$shape,
        GPD_LogLik = fit_q$loglik,
        Convergence = fit_q$convergence,
        FitMethod = fit_q$method,
        PrimaryThreshold = abs(q - EVT_THRESHOLD_PROB) < 1e-12
      )
    }
  }
  
  evt_threshold_stability <- if (length(stab_rows) > 0L) {
    rbindlist(stab_rows, use.names = TRUE, fill = TRUE)
  } else data.table()
  
  evt_pit_diagnostics <- if (length(pit_rows) > 0L) {
    rbindlist(pit_rows, use.names = TRUE, fill = TRUE)
  } else data.table()
  
  evt_parameter_uncertainty <- if (length(unc_rows) > 0L) {
    rbindlist(unc_rows, use.names = TRUE, fill = TRUE)
  } else data.table()
  
  # Reconstruct every scheduled refit attempt, including failed/fallback attempts.
  attempt_rows <- list()
  for (s in sort(unique(garch_evt_panel$Sector))) {
    pp <- garch_evt_panel[Sector == s][order(Date)]
    if (nrow(pp) <= GARCH_MIN_HISTORY) next
    attempt_idx <- seq.int(
      GARCH_MIN_HISTORY + 1L,
      nrow(pp),
      by = EVT_REFIT_EVERY
    )
    recorded_dates <- evt_parameter_history[Sector == s, RefitDate]
    
    for (ii in attempt_idx) {
      attempt_date <- pp$Date[ii]
      loss_hist <- -pp$StdInnovation[seq_len(ii - 1L)]
      loss_hist <- loss_hist[is.finite(loss_hist)]
      u <- as.numeric(stats::quantile(
        loss_hist,
        probs = EVT_THRESHOLD_PROB,
        na.rm = TRUE,
        names = FALSE,
        type = 8
      ))
      excess <- loss_hist[loss_hist > u] - u
      fit <- fit_gpd_mle(excess)
      accepted <- (
        is.finite(fit$scale) &&
          is.finite(fit$shape) &&
          fit$convergence == 0L &&
          fit$n_exceed >= EVT_MIN_EXCEEDANCES
      )
      prior_success <- any(recorded_dates < attempt_date)
      recorded_success <- any(recorded_dates == attempt_date)
      
      attempt_rows[[length(attempt_rows) + 1L]] <- data.table(
        Sector = s,
        AttemptDate = attempt_date,
        HistN = length(loss_hist),
        Threshold_u = u,
        ExceedN = length(excess),
        ReproducedAccepted = accepted,
        RecordedSuccessfulRefit = recorded_success,
        FallbackCarryForward = (!accepted && prior_success),
        LabelsUnavailable = (!accepted && !prior_success),
        FitMethod = fit$method,
        Convergence = fit$convergence
      )
    }
  }
  evt_refit_attempts <- if (length(attempt_rows) > 0L) {
    rbindlist(attempt_rows, use.names = TRUE, fill = TRUE)
  } else data.table()
  
  evt_refit_summary <- if (nrow(evt_refit_attempts) > 0L) {
    evt_refit_attempts[, .(
      ScheduledAttempts = .N,
      SuccessfulRefits = sum(RecordedSuccessfulRefit, na.rm = TRUE),
      FailedAttempts = sum(!RecordedSuccessfulRefit, na.rm = TRUE),
      FallbackCarryForwards = sum(FallbackCarryForward, na.rm = TRUE),
      LabelsUnavailableAttempts = sum(LabelsUnavailable, na.rm = TRUE),
      SuccessRate = mean(RecordedSuccessfulRefit, na.rm = TRUE),
      FallbackRate = mean(FallbackCarryForward, na.rm = TRUE)
    ), by = Sector]
  } else data.table()
  
  write_table_csv(
    evt_threshold_stability,
    "Table_R22_EVT_threshold_stability_MRL.csv"
  )
  write_table_csv(
    evt_pit_diagnostics,
    "Table_R23_EVT_PIT_tail_fit_diagnostics.csv"
  )
  write_table_csv(
    evt_parameter_uncertainty,
    "Table_R24_EVT_GPD_parameter_uncertainty.csv"
  )
  write_table_csv(
    evt_refit_attempts,
    "Table_R25_EVT_refit_attempts_and_fallbacks.csv"
  )
  write_table_csv(
    evt_refit_summary,
    "Table_R26_EVT_refit_success_fallback_summary.csv"
  )
}

# ------------------------------------------------------------------------------
# 16.1.5 Equal-weight and stricter-liquidity sector-return robustness
# ------------------------------------------------------------------------------
run_alternative_sector_evt <- function(panel_input, series_name) {
  panel_rows <- list()
  param_rows <- list()
  error_rows <- list()
  for (s in sort(unique(panel_input[PrimarySector == TRUE, Sector]))) {
    rr <- run_sector_garch_evt(s, panel_input)
    if (nrow(rr$panel) > 0L) {
      tmp <- copy(rr$panel)
      tmp[, RobustnessSeries := series_name]
      panel_rows[[length(panel_rows) + 1L]] <- tmp
    }
    if (nrow(rr$evt_params) > 0L) {
      tmp <- copy(rr$evt_params)
      tmp[, RobustnessSeries := series_name]
      param_rows[[length(param_rows) + 1L]] <- tmp
    }
    if (nrow(rr$error) > 0L) {
      tmp <- copy(rr$error)
      tmp[, RobustnessSeries := series_name]
      error_rows[[length(error_rows) + 1L]] <- tmp
    }
  }
  list(
    panel = if (length(panel_rows)) rbindlist(panel_rows, fill = TRUE) else data.table(),
    params = if (length(param_rows)) rbindlist(param_rows, fill = TRUE) else data.table(),
    errors = if (length(error_rows)) rbindlist(error_rows, fill = TRUE) else data.table()
  )
}

compare_tail_event_robustness <- function(primary_panel, alternative_panel, series_name) {
  if (nrow(primary_panel) == 0L || nrow(alternative_panel) == 0L) return(data.table())
  
  required_cols <- c(
    "Date", "Sector", "GARCH_Realized",
    "CrashThreshold_0025", "Crash_Main"
  )
  missing_primary <- setdiff(required_cols, names(primary_panel))
  missing_alternative <- setdiff(required_cols, names(alternative_panel))
  if (length(missing_primary) > 0L || length(missing_alternative) > 0L) {
    stop(
      paste0(
        "Robustness comparison cannot run because required columns are missing. ",
        "Primary missing: ", paste(missing_primary, collapse = ", "),
        "; Alternative missing: ", paste(missing_alternative, collapse = ", ")
      )
    )
  }
  
  p <- primary_panel[, .(
    Date, Sector,
    PrimaryReturn = GARCH_Realized,
    PrimaryThreshold = CrashThreshold_0025,
    PrimaryEvent = Crash_Main
  )]
  a <- alternative_panel[, .(
    Date, Sector,
    AlternativeReturn = GARCH_Realized,
    AlternativeThreshold = CrashThreshold_0025,
    AlternativeEvent = Crash_Main
  )]
  m <- merge(p, a, by = c("Date", "Sector"), all = FALSE)
  m <- m[!is.na(PrimaryEvent) & !is.na(AlternativeEvent)]
  if (nrow(m) == 0L) return(data.table())
  m[, RobustnessSeries := series_name]
  m[, .(
    NCommonLabelled = .N,
    PrimaryEvents = sum(PrimaryEvent == 1L),
    AlternativeEvents = sum(AlternativeEvent == 1L),
    LabelAgreement = mean(PrimaryEvent == AlternativeEvent),
    EventIntersection = sum(PrimaryEvent == 1L & AlternativeEvent == 1L),
    EventUnion = sum(PrimaryEvent == 1L | AlternativeEvent == 1L),
    EventJaccard = {
      u <- sum(PrimaryEvent == 1L | AlternativeEvent == 1L)
      if (u > 0L) sum(PrimaryEvent == 1L & AlternativeEvent == 1L) / u else NA_real_
    },
    ReturnCorrelation = suppressWarnings(stats::cor(PrimaryReturn, AlternativeReturn, use = "complete.obs")),
    ThresholdCorrelation = suppressWarnings(stats::cor(PrimaryThreshold, AlternativeThreshold, use = "complete.obs"))
  ), by = .(Sector, RobustnessSeries)]
}

equal_weight_evt <- list(panel = data.table(), params = data.table(), errors = data.table())
strict_liquidity_evt <- list(panel = data.table(), params = data.table(), errors = data.table())
robustness_event_comparison <- data.table()
strict_liquidity_coverage <- data.table()

if (RUN_EQUAL_WEIGHT_ROBUSTNESS) {
  log_msg("Reviewer robustness: running matched-support equal-weight GJR-GARCH/EVT series.")
  
  # Recompute equal-weight returns on exactly the same firm-level support used
  # by the primary lagged-market-cap portfolio. The core SectorReturn_EW column
  # is left unchanged so the frozen primary feature set is not altered.
  ew_matched_daily <- clean[, {
    ret_ok <- (
      is.finite(LogReturn) &
        is.finite(LagMarketCapKES) &
        LagMarketCapKES > 0
    )
    list(
      SectorReturn_EW_Matched = if (any(ret_ok)) {
        mean(LogReturn[ret_ok])
      } else NA_real_
    )
  }, by = .(Sector, Date)]
  
  ew_input <- merge(
    sector_panel,
    ew_matched_daily,
    by = c("Sector", "Date"),
    all.x = TRUE,
    sort = FALSE
  )
  ew_input[, SectorReturn_Model := fifelse(
    SectorDayEligible == 1L,
    SectorReturn_EW_Matched,
    NA_real_
  )]
  
  equal_weight_evt <- run_alternative_sector_evt(
    ew_input,
    "EqualWeight_MatchedSupport"
  )
  if (nrow(equal_weight_evt$panel) > 0L) {
    write_handoff(
      equal_weight_evt$panel,
      "robustness_equal_weight_garch_evt"
    )
    cmp <- compare_tail_event_robustness(
      garch_evt_panel,
      equal_weight_evt$panel,
      "EqualWeight_MatchedSupport"
    )
    robustness_event_comparison <- rbindlist(
      list(robustness_event_comparison, cmp),
      use.names = TRUE,
      fill = TRUE
    )
  }
}

if (RUN_STRICT_LIQUIDITY_ROBUSTNESS) {
  log_msg("Reviewer robustness: constructing strict-liquidity sector returns.")
  strict_daily <- clean[, {
    strict_ok <- (
      is.finite(LogReturn) &
        is.finite(LagMarketCapKES) &
        LagMarketCapKES > 0
    )
    if (ROBUST_REQUIRE_POSITIVE_VOLUME) {
      strict_ok <- strict_ok & is.finite(Volume) & Volume > 0
    }
    if (ROBUST_EXCLUDE_ZERO_RETURN) {
      strict_ok <- strict_ok & !is.na(ZeroReturn) & ZeroReturn == 0L
    }
    n_obs <- uniqueN(Stock)
    n_ret <- sum(strict_ok)
    share_ret <- if (n_obs > 0L) n_ret / n_obs else NA_real_
    w <- if (any(strict_ok)) {
      LagMarketCapKES[strict_ok] / sum(LagMarketCapKES[strict_ok])
    } else numeric(0)
    list(
      StrictNStocksObserved = n_obs,
      StrictNStocksReturn = n_ret,
      StrictConstituentReturnShare = share_ret,
      StrictSectorReturn_MCW = if (length(w)) sum(LogReturn[strict_ok] * w) else NA_real_
    )
  }, by = .(Sector, Date)]
  
  strict_daily[, StrictSectorDayEligible := as.integer(
    is.finite(StrictSectorReturn_MCW) &
      StrictNStocksReturn >= PRIMARY_MIN_DAILY_RETURNS &
      is.finite(StrictConstituentReturnShare) &
      StrictConstituentReturnShare >= PRIMARY_MIN_DAILY_RETURN_SHARE
  )]
  strict_daily[, StrictSectorReturn_Model := fifelse(
    StrictSectorDayEligible == 1L,
    StrictSectorReturn_MCW,
    NA_real_
  )]
  
  strict_liquidity_coverage <- strict_daily[, .(
    StrictEligibleDays = sum(StrictSectorDayEligible == 1L),
    StrictEligibleCoverage = mean(StrictSectorDayEligible == 1L),
    MedianStrictNStocksReturn = as.numeric(median(StrictNStocksReturn, na.rm = TRUE)),
    MedianStrictConstituentShare = as.numeric(median(StrictConstituentReturnShare, na.rm = TRUE))
  ), by = Sector]
  
  strict_input <- merge(
    sector_panel,
    strict_daily[, .(Sector, Date, StrictSectorReturn_Model)],
    by = c("Sector", "Date"),
    all.x = TRUE,
    sort = FALSE
  )
  strict_input[, SectorReturn_Model := StrictSectorReturn_Model]
  
  strict_liquidity_evt <- run_alternative_sector_evt(
    strict_input,
    "StrictPositiveVolume_NonZeroReturn"
  )
  if (nrow(strict_liquidity_evt$panel) > 0L) {
    write_handoff(
      strict_liquidity_evt$panel,
      "robustness_strict_liquidity_garch_evt"
    )
    cmp <- compare_tail_event_robustness(
      garch_evt_panel,
      strict_liquidity_evt$panel,
      "StrictPositiveVolume_NonZeroReturn"
    )
    robustness_event_comparison <- rbindlist(
      list(robustness_event_comparison, cmp),
      use.names = TRUE,
      fill = TRUE
    )
  }
  
  write_table_csv(
    strict_liquidity_coverage,
    "Table_R27_strict_liquidity_sector_coverage.csv"
  )
}

if (nrow(robustness_event_comparison) > 0L) {
  write_table_csv(
    robustness_event_comparison,
    "Table_R28_equal_weight_liquidity_tail_event_robustness.csv"
  )
}

# ------------------------------------------------------------------------------
# 16.1.6 Market-wide common-shock tail-event series for Hawkes robustness
# ------------------------------------------------------------------------------
market_common_shock_panel <- data.table()
market_common_shock_params <- data.table()

if (RUN_MARKET_COMMON_SHOCK_SERIES) {
  log_msg("Reviewer robustness: estimating NSE-wide market GJR-GARCH/EVT series.")
  market_input <- market_daily[, .(
    Date,
    Sector = "NSE_Market",
    PrimarySector = TRUE,
    SectorReturn_Model = MarketReturn_MCW
  )]
  market_res <- run_sector_garch_evt("NSE_Market", market_input)
  market_common_shock_panel <- market_res$panel
  market_common_shock_params <- market_res$evt_params
  
  if (nrow(market_res$error) > 0L) {
    log_msg(
      "WARNING: NSE-wide common-shock GARCH/EVT returned: ",
      paste(market_res$error$Error, collapse = " | ")
    )
    write_table_csv(
      market_res$error,
      "Table_R29B_market_common_shock_errors.csv"
    )
  }
  
  if (nrow(market_common_shock_panel) > 0L) {
    write_handoff(
      market_common_shock_panel,
      "market_common_shock_garch_evt"
    )
    write_table_csv(
      market_common_shock_panel[, .(
        Date,
        GARCH_Mu,
        GARCH_Sigma,
        StdInnovation,
        EVT_u,
        EVT_scale,
        EVT_shape,
        CrashThreshold_0025,
        Crash_Main
      )],
      "Table_R29_market_common_shock_tail_event_series.csv"
    )
  }
  if (nrow(market_common_shock_params) > 0L) {
    write_handoff(
      market_common_shock_params,
      "market_common_shock_evt_parameter_history"
    )
  }
}

# ------------------------------------------------------------------------------
# 16.1.7 Primary-output integrity comparison against frozen v2 handoff
# ------------------------------------------------------------------------------
primary_integrity_comparison <- data.table(
  ReferenceFile = REFERENCE_PRIMARY_HANDOFF,
  ReferenceFound = file.exists(REFERENCE_PRIMARY_HANDOFF),
  ExpectedReferenceRows = NA_integer_,
  CommonRows = NA_integer_,
  RowCoverage = NA_real_,
  MaxAbsSectorReturnDifference = NA_real_,
  SectorReturnNAPatternMismatches = NA_integer_,
  PrimaryEventMismatches = NA_integer_,
  PrimaryEventNAPatternMismatches = NA_integer_,
  ThresholdMismatches = NA_integer_,
  ThresholdNAPatternMismatches = NA_integer_,
  Passed = NA
)

if (file.exists(REFERENCE_PRIMARY_HANDOFF)) {
  old_primary <- data.table::fread(
    REFERENCE_PRIMARY_HANDOFF,
    na.strings = c("", "NA", "N/A", "NULL", "null")
  )
  data.table::setDT(old_primary)
  old_primary[, Date := as.Date(Date)]
  
  old_threshold_col <- if ("CrashThreshold_Main" %in% names(old_primary)) {
    "CrashThreshold_Main"
  } else if ("CrashThreshold_0025" %in% names(old_primary)) {
    "CrashThreshold_0025"
  } else {
    NA_character_
  }
  
  base_needed <- c("Date", "Sector", "SectorReturn_Model", "Crash_Main")
  
  if (all(base_needed %in% names(old_primary)) && !is.na(old_threshold_col)) {
    old_small <- data.table::copy(
      old_primary[, c(base_needed, old_threshold_col), with = FALSE]
    )
    
    data.table::setnames(
      old_small,
      old_threshold_col,
      "OldCrashThreshold_Main"
    )
    data.table::setnames(
      old_small,
      c("SectorReturn_Model", "Crash_Main"),
      c("OldSectorReturn_Model", "OldCrash_Main")
    )
    
    new_small <- final_panel[, .(
      Date,
      Sector,
      SectorReturn_Model,
      CrashThreshold_Main,
      Crash_Main
    )]
    
    cmp <- merge(
      new_small,
      old_small,
      by = c("Date", "Sector"),
      all = FALSE
    )
    
    expected_reference_rows <- nrow(old_small)
    common_rows <- nrow(cmp)
    row_coverage <- if (expected_reference_rows > 0L) {
      common_rows / expected_reference_rows
    } else NA_real_
    
    return_na_mismatch <- xor(
      is.na(cmp$SectorReturn_Model),
      is.na(cmp$OldSectorReturn_Model)
    )
    event_na_mismatch <- xor(
      is.na(cmp$Crash_Main),
      is.na(cmp$OldCrash_Main)
    )
    threshold_na_mismatch <- xor(
      is.na(cmp$CrashThreshold_Main),
      is.na(cmp$OldCrashThreshold_Main)
    )
    
    return_diff <- abs(
      cmp$SectorReturn_Model - cmp$OldSectorReturn_Model
    )
    threshold_diff <- abs(
      cmp$CrashThreshold_Main - cmp$OldCrashThreshold_Main
    )
    
    event_value_mismatch <- (
      !is.na(cmp$Crash_Main) &
        !is.na(cmp$OldCrash_Main) &
        cmp$Crash_Main != cmp$OldCrash_Main
    )
    threshold_value_mismatch <- (
      is.finite(threshold_diff) &
        threshold_diff > 1e-12
    )
    
    max_return_diff <- if (any(is.finite(return_diff))) {
      max(return_diff, na.rm = TRUE)
    } else 0
    
    n_return_na_mismatch <- sum(return_na_mismatch, na.rm = TRUE)
    n_event_mismatch <- sum(event_value_mismatch, na.rm = TRUE)
    n_event_na_mismatch <- sum(event_na_mismatch, na.rm = TRUE)
    n_threshold_mismatch <- sum(threshold_value_mismatch, na.rm = TRUE)
    n_threshold_na_mismatch <- sum(threshold_na_mismatch, na.rm = TRUE)
    
    passed <- (
      expected_reference_rows > 0L &&
        common_rows == expected_reference_rows &&
        isTRUE(all.equal(row_coverage, 1, tolerance = 0)) &&
        max_return_diff <= 1e-12 &&
        n_return_na_mismatch == 0L &&
        n_event_mismatch == 0L &&
        n_event_na_mismatch == 0L &&
        n_threshold_mismatch == 0L &&
        n_threshold_na_mismatch == 0L
    )
    
    primary_integrity_comparison <- data.table(
      ReferenceFile = REFERENCE_PRIMARY_HANDOFF,
      ReferenceFound = TRUE,
      ExpectedReferenceRows = expected_reference_rows,
      CommonRows = common_rows,
      RowCoverage = row_coverage,
      MaxAbsSectorReturnDifference = max_return_diff,
      SectorReturnNAPatternMismatches = n_return_na_mismatch,
      PrimaryEventMismatches = n_event_mismatch,
      PrimaryEventNAPatternMismatches = n_event_na_mismatch,
      ThresholdMismatches = n_threshold_mismatch,
      ThresholdNAPatternMismatches = n_threshold_na_mismatch,
      Passed = passed
    )
    
    # Write failure evidence before stopping.
    write_table_csv(
      primary_integrity_comparison,
      "Table_R30_primary_output_integrity_vs_v2.csv"
    )
    
    if (!passed) {
      stop(
        "PRIMARY INTEGRITY FAILURE: reviewer-revision R code changed the frozen ",
        "market-cap-weighted primary return/2.5% event construction or row/NA ",
        "coverage. Review Table_R30_primary_output_integrity_vs_v2.csv."
      )
    }
  } else {
    log_msg(
      "WARNING: Reference v2 handoff found but lacks one or more ",
      "integrity-comparison columns."
    )
  }
} else {
  log_msg(
    "WARNING: Frozen v2 handoff was not found at: ",
    REFERENCE_PRIMARY_HANDOFF,
    ". Primary integrity comparison could not be executed automatically."
  )
}

write_table_csv(
  primary_integrity_comparison,
  "Table_R30_primary_output_integrity_vs_v2.csv"
)

# ------------------------------------------------------------------------------
# 16.1.8 Reviewer-diagnostic figures
# ------------------------------------------------------------------------------
if (nrow(evt_threshold_stability) > 0L) {
  mrl_plot_data <- evt_threshold_stability[, .(
    MedianMRL = median(MeanResidualLife, na.rm = TRUE),
    Q25MRL = as.numeric(quantile(MeanResidualLife, 0.25, na.rm = TRUE, names = FALSE)),
    Q75MRL = as.numeric(quantile(MeanResidualLife, 0.75, na.rm = TRUE, names = FALSE))
  ), by = .(Sector, ThresholdProb)]
  
  fig_mrl <- ggplot(
    mrl_plot_data,
    aes(x = ThresholdProb, y = MedianMRL)
  ) +
    geom_ribbon(aes(ymin = Q25MRL, ymax = Q75MRL), alpha = 0.18) +
    geom_line(linewidth = 0.5) +
    geom_point(size = 1.2) +
    facet_wrap(~Sector, scales = "free_y", ncol = 2) +
    labs(
      x = "EVT threshold probability",
      y = "Mean residual life (standardised-loss units)",
      title = "EVT mean-residual-life stability across thresholds"
    ) +
    theme_minimal(base_size = 10)
  save_plot_both(fig_mrl, "Figure_R06_EVT_mean_residual_life_stability", width = 12, height = 10)
  
  shape_plot_data <- evt_threshold_stability[
    is.finite(GPD_Shape),
    .(
      MedianShape = median(GPD_Shape, na.rm = TRUE),
      Q25Shape = as.numeric(quantile(GPD_Shape, 0.25, na.rm = TRUE, names = FALSE)),
      Q75Shape = as.numeric(quantile(GPD_Shape, 0.75, na.rm = TRUE, names = FALSE))
    ),
    by = .(Sector, ThresholdProb)
  ]
  if (nrow(shape_plot_data) > 0L) {
    fig_shape <- ggplot(
      shape_plot_data,
      aes(x = ThresholdProb, y = MedianShape)
    ) +
      geom_ribbon(aes(ymin = Q25Shape, ymax = Q75Shape), alpha = 0.18) +
      geom_line(linewidth = 0.5) +
      geom_point(size = 1.2) +
      facet_wrap(~Sector, scales = "free_y", ncol = 2) +
      labs(
        x = "EVT threshold probability",
        y = "GPD shape parameter",
        title = "GPD shape stability across thresholds"
      ) +
      theme_minimal(base_size = 10)
    save_plot_both(fig_shape, "Figure_R07_EVT_shape_threshold_stability", width = 12, height = 10)
  }
}

if (nrow(tail_event_rows) > 0L) {
  fig_magnitude <- ggplot(
    tail_event_rows,
    aes(x = Sector, y = SectorReturn)
  ) +
    geom_boxplot(outlier.alpha = 0.35) +
    coord_flip() +
    scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    labs(
      x = NULL,
      y = "Realised sector return on primary tail-event days",
      title = "Economic magnitude of volatility-conditioned sector tail events"
    ) +
    theme_minimal(base_size = 10)
  save_plot_both(fig_magnitude, "Figure_R08_primary_tail_event_return_magnitudes", width = 10, height = 7)
}

# ------------------------------------------------------------------------------
# 16.1.9 Reviewer-revision diagnostic workbook and handoff bundle
# ------------------------------------------------------------------------------
wb_reviewer <- createWorkbook()
add_excel_sheet(wb_reviewer, "Revision phases", revision_phase_counts)
add_excel_sheet(wb_reviewer, "Constituent history", observed_constituent_history)
add_excel_sheet(wb_reviewer, "Annual composition", sector_composition_yearly)
add_excel_sheet(wb_reviewer, "Tail-event magnitudes", tail_event_magnitude_summary)
add_excel_sheet(wb_reviewer, "Tail-event rows", tail_event_rows)
if (nrow(evt_threshold_stability) > 0L) {
  add_excel_sheet(wb_reviewer, "EVT threshold stability", evt_threshold_stability)
}
if (nrow(evt_pit_diagnostics) > 0L) {
  add_excel_sheet(wb_reviewer, "EVT PIT diagnostics", evt_pit_diagnostics)
}
if (nrow(evt_parameter_uncertainty) > 0L) {
  add_excel_sheet(wb_reviewer, "EVT uncertainty", evt_parameter_uncertainty)
}
if (nrow(evt_refit_summary) > 0L) {
  add_excel_sheet(wb_reviewer, "EVT fallback summary", evt_refit_summary)
}
if (nrow(robustness_event_comparison) > 0L) {
  add_excel_sheet(wb_reviewer, "Weight liquidity robust", robustness_event_comparison)
}
if (nrow(strict_liquidity_coverage) > 0L) {
  add_excel_sheet(wb_reviewer, "Strict liquidity coverage", strict_liquidity_coverage)
}
add_excel_sheet(wb_reviewer, "Primary integrity", primary_integrity_comparison)

saveWorkbook(
  wb_reviewer,
  file.path(DIR_EXCEL, "04_Reviewer_Revision_Diagnostics.xlsx"),
  overwrite = TRUE
)

# Handoff diagnostic tables required by later Hawkes / Python reviewer stages.
write_handoff(revision_phase_counts, "reviewer_revision_phase_counts")
if (nrow(evt_refit_summary) > 0L) {
  write_handoff(evt_refit_summary, "evt_refit_success_fallback_summary")
}
if (nrow(robustness_event_comparison) > 0L) {
  write_handoff(robustness_event_comparison, "sector_return_robustness_summary")
}

reviewer_revision_metadata <- data.table(
  Key = c(
    "InnerTuneStart",
    "TrainEnd",
    "CalibrationEnd",
    "TestStart",
    "ValidationLegacyRole",
    "EVTWindow",
    "EVTDiagnosticThresholdProbabilities",
    "EVTDiagnosticRefitStride",
    "EqualWeightRobustness",
    "StrictLiquidityRobustness",
    "MarketCommonShockSeries",
    "DataSourceNote",
    "DataLicenseNote"
  ),
  Value = c(
    as.character(INNER_TUNE_START),
    as.character(TRAIN_END),
    as.character(VALID_END),
    as.character(VALID_END + 1L),
    "CalibrationOnly_in_revised_Python_design",
    "Expanding_history_strictly_before_forecast_date",
    paste(EVT_DIAGNOSTIC_THRESHOLD_PROBS, collapse = ","),
    EVT_DIAGNOSTIC_REFIT_STRIDE,
    RUN_EQUAL_WEIGHT_ROBUSTNESS,
    RUN_STRICT_LIQUIDITY_ROBUSTNESS,
    RUN_MARKET_COMMON_SHOCK_SERIES,
    DATA_SOURCE_NOTE,
    DATA_LICENSE_NOTE
  )
)

write_handoff(reviewer_revision_metadata, "reviewer_revision_metadata")
write_table_csv(
  reviewer_revision_metadata,
  "Table_R31_reviewer_revision_metadata.csv"
)

# ==============================================================================
# 17. EXCEL WORKBOOKS
# ==============================================================================

log_msg("Writing Excel workbooks.")

# Workbook 1: Data audit.
wb_audit <- createWorkbook()
add_excel_sheet(wb_audit, "Overview", audit_overview)
add_excel_sheet(wb_audit, "Missingness", missingness_raw)
add_excel_sheet(wb_audit, "Sector composition", sector_composition)
add_excel_sheet(wb_audit, "Sector coverage", sector_coverage)
add_excel_sheet(wb_audit, "Thin trading by stock", stock_thin_trading)
add_excel_sheet(wb_audit, "Corporate actions", corporate_action_table)

if (nrow(critical_rows) > 0L) {
  add_excel_sheet(wb_audit, "Excluded critical rows", critical_rows)
}
if (nrow(invalid_ohlc_rows) > 0L) {
  add_excel_sheet(wb_audit, "Invalid OHLC retained", invalid_ohlc_rows)
}

saveWorkbook(
  wb_audit,
  file.path(DIR_EXCEL, "01_Data_Audit_and_Sector_Coverage.xlsx"),
  overwrite = TRUE
)

# Workbook 2: Descriptive and econometric outputs.
wb_results <- createWorkbook()
add_excel_sheet(wb_results, "Sector descriptive", sector_summary)

if (nrow(garch_candidate_results) > 0L) {
  add_excel_sheet(wb_results, "GARCH comparison", garch_candidate_results)
}
if (exists("evt_summary") && nrow(evt_summary) > 0L) {
  add_excel_sheet(wb_results, "EVT summary", evt_summary)
}
if (nrow(crash_counts) > 0L) {
  add_excel_sheet(wb_results, "Crash counts split", crash_counts)
}
if (nrow(crash_counts_tail) > 0L) {
  add_excel_sheet(wb_results, "Crash tails", crash_counts_tail)
}
if (nrow(garch_evt_errors) > 0L) {
  add_excel_sheet(wb_results, "Errors", garch_evt_errors)
}

saveWorkbook(
  wb_results,
  file.path(DIR_EXCEL, "02_GARCH_EVT_and_Crash_Results.xlsx"),
  overwrite = TRUE
)

# Workbook 3: Python handoff preview (not the full high-volume data if it is too
# large for comfortable Excel use).
wb_preview <- createWorkbook()
preview_cols <- intersect(
  c(
    "Date", "Sector", "PrimarySector", "Split",
    "NStocksObserved", "NStocksReturn", "ConstituentReturnShare",
    "SectorDayEligible", "EligibleDayCoverage",
    "SectorReturn_MCW", "SectorReturn_Model", "ParkinsonVol", "GarmanKlassVol",
    "Turnover_MCW", "AmihudILLIQ_Median", "BreadthNegative",
    "ZeroReturnShare", "ReturnDispersion", "HHI",
    "GARCH_Sigma", "StdInnovation",
    "CrashThreshold_Main", "Crash_Main"
  ),
  names(final_panel)
)
preview_data <- final_panel[, ..preview_cols]
preview_data <- head(preview_data, 10000L)
add_excel_sheet(
  wb_preview,
  "Panel preview",
  preview_data
)
add_excel_sheet(wb_preview, "Sector coverage", sector_coverage)
if (nrow(crash_counts) > 0L) {
  add_excel_sheet(wb_preview, "Crash counts", crash_counts)
}

saveWorkbook(
  wb_preview,
  file.path(DIR_EXCEL, "03_Python_Handoff_Preview.xlsx"),
  overwrite = TRUE
)

# ==============================================================================
# 18. PYTHON/COLAB HANDOFF FILES
# ==============================================================================

log_msg("Writing frozen Python handoff datasets.")

# 18.1 Main sector modelling panel.
write_handoff(
  final_panel,
  "sector_forecasting_master"
)

# 18.2 Stock-level clean panel for optional Python robustness work.
stock_handoff_cols <- intersect(
  c(
    "Date", "TradingIndex", "Sector", "Stock",
    "Open", "High", "Low", "Close", "Volume",
    "IssuedShares", "MarketCapKES", "LagMarketCapKES",
    "TradingGap", "CorporateActionFlag", "CorporateActionWindow3",
    "LogReturn", "LogReturn_CA3",
    "ParkinsonVar", "GarmanKlassVar", "RogersSatchellVar",
    "Turnover", "TradedValueKES", "AmihudILLIQ",
    "ZeroReturn", "ZeroVolume", "NegativeReturn"
  ),
  names(clean)
)

write_handoff(
  clean[, ..stock_handoff_cols],
  "stock_clean_internal_features"
)

# 18.3 GARCH-EVT refit history.
if (nrow(evt_parameter_history) > 0L) {
  write_handoff(
    evt_parameter_history,
    "evt_parameter_history"
  )
}

# 18.4 Metadata/configuration.
metadata <- data.table(
  Key = c(
    "ProjectName",
    "InputFile",
    "DataStart",
    "DataEnd",
    "UniqueStocks",
    "UniqueSectors",
    "PrimaryMinStocks",
    "PrimaryMinDailyReturns",
    "PrimaryMinDailyReturnShare",
    "PrimaryMinEligibleDayCoverage",
    "CorporateActionShareTolerance",
    "GARCHMainModel",
    "GARCHMainDistribution",
    "GARCHMinHistory",
    "GARCHRefitEvery",
    "EVTThresholdProbability",
    "EVTMainTailProbability",
    "EVTTailProbabilities",
    "EVTRefitEvery",
    "EVTMinExceedances",
    "TrainEnd",
    "ValidationEnd",
    "InnerTuneStart",
    "ValidationRoleInRevision",
    "ForecastHorizons",
    "RandomSeed",
    "ScriptVersion"
  ),
  Value = c(
    PROJECT_NAME,
    basename(INPUT_FILE),
    as.character(min(clean$Date, na.rm = TRUE)),
    as.character(max(clean$Date, na.rm = TRUE)),
    uniqueN(clean$Stock),
    uniqueN(clean$Sector),
    PRIMARY_MIN_STOCKS,
    PRIMARY_MIN_DAILY_RETURNS,
    PRIMARY_MIN_DAILY_RETURN_SHARE,
    PRIMARY_MIN_ELIGIBLE_DAY_COVERAGE,
    SHARE_CHANGE_TOL,
    GARCH_MAIN_MODEL,
    GARCH_MAIN_DIST,
    GARCH_MIN_HISTORY,
    GARCH_REFIT_EVERY,
    EVT_THRESHOLD_PROB,
    EVT_MAIN_TAIL_PROB,
    paste(EVT_TAIL_PROBS, collapse = ","),
    EVT_REFIT_EVERY,
    EVT_MIN_EXCEEDANCES,
    as.character(TRAIN_END),
    as.character(VALID_END),
    as.character(INNER_TUNE_START),
    "CalibrationOnly",
    paste(FORECAST_HORIZONS, collapse = ","),
    RANDOM_SEED,
    "3.4-reviewer-revision-final-audited"
  )
)

write_handoff(metadata, "study_metadata")
write_table_csv(metadata, "Table_R15_study_metadata.csv")

# 18.5 Save R-native checkpoint for debugging/reproducibility only.
saveRDS(
  list(
    sector_forecasting_master = final_panel,
    sector_coverage = sector_coverage,
    evt_parameter_history = evt_parameter_history,
    garch_candidate_results = garch_candidate_results,
    crash_counts = crash_counts,
    reviewer_revision_phase_counts = revision_phase_counts,
    evt_threshold_stability = evt_threshold_stability,
    evt_pit_diagnostics = evt_pit_diagnostics,
    evt_parameter_uncertainty = evt_parameter_uncertainty,
    evt_refit_summary = evt_refit_summary,
    robustness_event_comparison = robustness_event_comparison,
    market_common_shock_panel = market_common_shock_panel,
    primary_integrity_comparison = primary_integrity_comparison
  ),
  file.path(DIR_HANDOFF, "R_phase_checkpoint.rds"),
  compress = "xz"
)

# ==============================================================================
# 18.6 FINAL REVIEWER-REVISION OUTPUT GUARD
# ==============================================================================
required_reviewer_files <- c(
  file.path(DIR_TABLES, "Table_R16_reviewer_revision_phase_counts.csv"),
  file.path(DIR_TABLES, "Table_R20_primary_tail_event_observations.csv"),
  file.path(DIR_TABLES, "Table_R21_primary_tail_event_magnitude_summary.csv"),
  file.path(DIR_TABLES, "Table_R30_primary_output_integrity_vs_v2.csv"),
  file.path(DIR_TABLES, "Table_R31_reviewer_revision_metadata.csv"),
  file.path(DIR_EXCEL, "04_Reviewer_Revision_Diagnostics.xlsx")
)

if (RUN_REVIEWER_DIAGNOSTICS) {
  required_reviewer_files <- c(
    required_reviewer_files,
    file.path(DIR_TABLES, "Table_R22_EVT_threshold_stability_MRL.csv"),
    file.path(DIR_TABLES, "Table_R23_EVT_PIT_tail_fit_diagnostics.csv"),
    file.path(DIR_TABLES, "Table_R24_EVT_GPD_parameter_uncertainty.csv"),
    file.path(DIR_TABLES, "Table_R25_EVT_refit_attempts_and_fallbacks.csv"),
    file.path(DIR_TABLES, "Table_R26_EVT_refit_success_fallback_summary.csv")
  )
}
if (RUN_EQUAL_WEIGHT_ROBUSTNESS || RUN_STRICT_LIQUIDITY_ROBUSTNESS) {
  required_reviewer_files <- c(
    required_reviewer_files,
    file.path(DIR_TABLES, "Table_R28_equal_weight_liquidity_tail_event_robustness.csv")
  )
}
if (RUN_EQUAL_WEIGHT_ROBUSTNESS) {
  required_reviewer_files <- c(
    required_reviewer_files,
    file.path(DIR_HANDOFF, "robustness_equal_weight_garch_evt.csv.gz")
  )
}
if (RUN_STRICT_LIQUIDITY_ROBUSTNESS) {
  required_reviewer_files <- c(
    required_reviewer_files,
    file.path(DIR_TABLES, "Table_R27_strict_liquidity_sector_coverage.csv"),
    file.path(DIR_HANDOFF, "robustness_strict_liquidity_garch_evt.csv.gz")
  )
}
if (RUN_MARKET_COMMON_SHOCK_SERIES) {
  required_reviewer_files <- c(
    required_reviewer_files,
    file.path(DIR_TABLES, "Table_R29_market_common_shock_tail_event_series.csv")
  )
}

missing_reviewer_files <- required_reviewer_files[
  !file.exists(required_reviewer_files)
]

if (length(missing_reviewer_files) > 0L) {
  stop(
    paste0(
      "Reviewer-revision output guard failed. Missing files: ",
      paste(basename(missing_reviewer_files), collapse = " | ")
    )
  )
}

# ==============================================================================
# 19. SESSION INFORMATION, FILE MANIFEST, AND MD5 HASHES
# ==============================================================================

session_file <- file.path(DIR_LOGS, paste0("sessionInfo_", RUN_STAMP, ".txt"))
capture.output(sessionInfo(), file = session_file)

# Manifest excludes ZIP files to avoid self-reference.
manifest_files <- list.files(
  OUTPUT_ROOT,
  recursive = TRUE,
  full.names = TRUE
)
manifest_files_norm <- gsub("\\\\", "/", manifest_files)
manifest_files <- manifest_files[!grepl("/06_Zip/", manifest_files_norm, fixed = TRUE)]

manifest <- data.table(
  RelativePath = substring(
    normalizePath(manifest_files, winslash = "/", mustWork = FALSE),
    nchar(normalizePath(OUTPUT_ROOT, winslash = "/", mustWork = FALSE)) + 2L
  ),
  SizeBytes = file.info(manifest_files)$size,
  Modified = as.character(file.info(manifest_files)$mtime),
  MD5 = unname(tools::md5sum(manifest_files))
)

data.table::fwrite(
  manifest,
  file.path(DIR_LOGS, paste0("output_manifest_", RUN_STAMP, ".csv"))
)

# ==============================================================================
# 20. ZIP ALL R OUTPUTS
# ==============================================================================

log_msg("Creating ZIP archive.")

zip_inputs <- c(
  list.files(DIR_TABLES, full.names = TRUE, recursive = TRUE),
  list.files(DIR_FIGURES, full.names = TRUE, recursive = TRUE),
  list.files(DIR_EXCEL, full.names = TRUE, recursive = TRUE),
  list.files(DIR_HANDOFF, full.names = TRUE, recursive = TRUE),
  list.files(DIR_LOGS, full.names = TRUE, recursive = TRUE)
)

zip_inputs <- zip_inputs[file.exists(zip_inputs)]

ZIP_FILE <- file.path(
  DIR_ZIP,
  paste0(PROJECT_NAME, "_R_Outputs_", RUN_STAMP, ".zip")
)

old_wd <- getwd()
root_norm <- normalizePath(OUTPUT_ROOT, winslash = "/", mustWork = TRUE)
zip_inputs_norm <- normalizePath(zip_inputs, winslash = "/", mustWork = TRUE)
zip_rel <- substring(zip_inputs_norm, nchar(root_norm) + 2L)

setwd(OUTPUT_ROOT)
zip::zipr(
  zipfile = ZIP_FILE,
  files = zip_rel,
  recurse = TRUE,
  include_directories = FALSE
)
setwd(old_wd)

# ==============================================================================
# 21. FINAL CONSOLE SUMMARY
# ==============================================================================

log_msg("R phase completed successfully.")
log_msg("Tables folder:", DIR_TABLES)
log_msg("Figures folder:", DIR_FIGURES)
log_msg("Excel folder:", DIR_EXCEL)
log_msg("Python handoff folder:", DIR_HANDOFF)
log_msg("ZIP archive:", ZIP_FILE)

cat("\n")
cat("===============================================================================\n")
cat("R PHASE COMPLETE\n")
cat("===============================================================================\n")
cat("Input file:\n  ", INPUT_FILE, "\n\n", sep = "")
cat("Output root:\n  ", OUTPUT_ROOT, "\n\n", sep = "")
cat("Python handoff:\n  ", DIR_HANDOFF, "\n\n", sep = "")
cat("ZIP archive:\n  ", ZIP_FILE, "\n\n", sep = "")

if ("Crash_Main" %in% names(final_panel)) {
  cat("Primary EVT tail-event probability (legacy Crash_Main):", EVT_MAIN_TAIL_PROB, "\n")
  cat("Primary sectors:\n")
  print(sector_coverage[PrimarySector == TRUE, .(
    Sector, NStocks, ReturnCoverage, EligibleDayCoverage
  )])
  cat("\nPrimary tail-event counts by legacy split:\n")
  print(crash_counts)
}

cat("\nIMPORTANT NEXT STEP:\n")
cat(
  "Review the primary event counts plus Tables R22-R31 reviewer diagnostics ",
  "before running the revised Hawkes/Python stages.\n",
  sep = ""
)
cat("Do NOT redefine the primary tail-event label in Python. Keep legacy Crash_Main for compatibility.\n")
cat("===============================================================================\n")

