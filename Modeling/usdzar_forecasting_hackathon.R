# USDZAR 4 Week forecast
# ============================================================================

# libraries
required <- c("tidyverse", "forecast", "ggplot2", "scales", "zoo", "patchwork", "knitr", "grid", "gridExtra" , "sandwich", "lmtest")
missing <- setdiff(required, rownames(installed.packages()))
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(tidyverse); library(forecast); library(ggplot2)
  library(scales); library(zoo); library(patchwork); library(knitr); library(sandwich); library(lmtest)
})

set.seed(42) # reproducible bootstrap
DATA_DIR <- "."
OUT_DIR  <- "outputs"
dir.create(OUT_DIR, showWarnings = FALSE)

# ---- 1. loaders ------------------------------------------------------------
read_wf <- function(file, new_name) {
  read_csv(file.path(DATA_DIR, file), show_col_types = FALSE) %>%
    transmute(date = as.Date(date), !!new_name := as.numeric(value))
}

fx <- read_wf("USDZAR_weekly_friday.csv",  "usdzar")
repo_sa <- read_wf("REPO_SA_weekly_friday.csv", "sa_repo")
fed_ffr <- read_wf("FED_Rate_weekly_friday.csv","us_ffr")
cpi_sa <- read_wf("CPI_SA_weekly_friday.csv",  "cpi_sa")
cpi_us <- read_wf("CPI_US_weekly_friday.csv",  "cpi_us")

platinum <- read_csv(file.path(DATA_DIR, "Platinum_Futures_Historical_Data.csv"), show_col_types = FALSE) %>%
  transmute(date  = as.Date(Date, format = "%Y/%m/%d") - 2,   # Sun close -> Fri
            price = as.numeric(gsub(",", "", Price))) %>%
  arrange(date)

vix_daily <- read_csv(file.path(DATA_DIR, "VIXCLS.csv"), show_col_types = FALSE) %>%
  transmute(date = as.Date(observation_date), vix = as.numeric(VIXCLS)) %>%
  filter(!is.na(vix)) %>%
  arrange(date)

vix_friday <- tibble(date = fx$date) %>% # the same Friday grid as USDZAR
  left_join(vix_daily, by = "date") %>%
  mutate(vix = na.locf(vix, na.rm = FALSE)) # holiday Friday -> prior close

#2. panel assembly (friday grid)
panel <- fx %>%
  left_join(repo_sa, by = "date") %>%
  left_join(fed_ffr, by = "date") %>%
  left_join(cpi_sa, by = "date") %>%
  left_join(cpi_us, by = "date") %>%
  left_join(platinum,by = "date") %>%
  left_join(vix_friday, by = "date") %>%
  arrange(date) %>%
  mutate(
    sa_repo = na.locf(sa_repo, na.rm = FALSE),
    us_ffr = na.locf(us_ffr, na.rm = FALSE),
    cpi_sa = na.locf(cpi_sa, na.rm = FALSE),
    cpi_us = na.locf(cpi_us, na.rm = FALSE),
    price = na.locf(price, na.rm = FALSE),

    rate_diff = sa_repo - us_ffr,
    cpi_sa_yoy = log(cpi_sa) - log(lag(cpi_sa, 52)),
    cpi_us_yoy = log(cpi_us) - log(lag(cpi_us, 52)),
    infl_diff = cpi_sa_yoy - cpi_us_yoy,
    dlog_platinum = log(price) - log(lag(price, 1)),
    dlog_vix = log(vix) - log(lag(vix, 1))
  ) %>%
  filter(if_all(c(rate_diff, infl_diff, dlog_platinum), ~ !is.na(.))) %>%
  select(date, usdzar, rate_diff, infl_diff, dlog_platinum, dlog_vix, vix, sa_repo, us_ffr, price)

stopifnot("strictly increasing fridays" = all(diff(panel$date) == 7) && all(weekdays(panel$date) == "Friday"),
  "no NAs in final panel" = !any(is.na(panel %>% select(-price))),
  "eval window covers 2021-2025" = min(panel$date) <= as.Date("2021-01-01") && max(panel$date) >= as.Date("2025-12-01"))
cat(sprintf("Panel: %d Fridays, %s to %s\n", nrow(panel), min(panel$date), max(panel$date)))

# 3. summary statistics table
# ============================================================================
summary_tbl <- panel %>%
  filter(date >= as.Date("2009-01-01")) %>%
  select(usdzar, rate_diff, infl_diff, dlog_platinum, dlog_vix) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "x") %>%
  group_by(variable) %>%
  summarise(n = n(), mean = mean(x), median = median(x), sd = sd(x),
            min = min(x), max = max(x), .groups = "drop") %>%
  mutate(across(mean:max, ~ round(., 4)))
write_csv(summary_tbl, file.path(OUT_DIR, "summary_statistics.csv"))

# rendered PNG table for the presentation 
p_stats <- summary_tbl %>%
  gridExtra::tableGrob(rows = NULL, theme = gridExtra::ttheme_minimal())
ggsave(file.path(OUT_DIR, "summary_statistics.png"), p_stats, width = 9, height = 2.5, dpi = 200)

# 4. rolling forecast loop
# ============================================================================
H <- 4
xreg_full <- panel %>%
  transmute(date, rate_diff_lag = lag(rate_diff, 1), infl_diff_lag = lag(infl_diff, 1),
            dlog_platinum_lag = lag(dlog_platinum, 1), dlog_vix = dlog_vix) %>%        # (note) unlagged 
  filter(if_all(-date, ~ !is.na(.)))

panel_use <- panel %>% select(-dlog_vix) %>% inner_join(xreg_full, by = "date")

vars3 <- c("rate_diff_lag", "infl_diff_lag", "dlog_platinum_lag")
vars4 <- c(vars3, "dlog_vix")

origins <- panel_use$date[panel_use$date >= as.Date("2021-01-01") & panel_use$date <= (max(panel_use$date) - 7 * H)]

results <- vector("list", length(origins))
for (i in seq_along(origins)) {
  origin <- origins[i]
  target_date <- origin + 7 * H
  train <- panel_use %>% filter(date < origin)
  stopifnot(nrow(train) > 100)
  
  y_train <- log(train$usdzar)
  x3_tr <- as.matrix(train[, vars3])
  x4_tr <- as.matrix(train[, vars4])
  
  # freeze last observed regressor row across the 4-week horizon
  # for dlog_vix this injects the origin-week risk shock into the forecast path
  x3_fut <- matrix(rep(tail(x3_tr, 1), H), nrow = H, byrow = TRUE); colnames(x3_fut) <- vars3
  x4_fut <- matrix(rep(tail(x4_tr, 1), H), nrow = H, byrow = TRUE); colnames(x4_fut) <- vars4
  
  fit_ar1 <- tryCatch(Arima(y_train, order = c(1,0,0)), error = function(e) NULL)
  fit_arx <- tryCatch(Arima(y_train, order = c(1,0,0), xreg = x3_tr), error = function(e) NULL)
  fit_vix <- tryCatch(Arima(y_train, order = c(1,0,0), xreg = x4_tr), error = function(e) NULL)
  if (is.null(fit_ar1) || is.null(fit_arx) || is.null(fit_vix)) next
  
  fc_ar1_log <- as.numeric(forecast(fit_ar1, h = H)$mean)[H]
  fc_arx_log <- as.numeric(forecast(fit_arx, h = H, xreg = x3_fut)$mean)[H]
  fc_vix_log <- as.numeric(forecast(fit_vix, h = H, xreg = x4_fut)$mean)[H]
  
  actual_row <- panel_use %>% filter(date == target_date)
  if (nrow(actual_row) == 0) next
  
  results[[i]] <- tibble(forecast_date = origin, target_date = target_date,
    model = c("ar1", "arimax", "arimax_vix"),
    forecast = exp(c(fc_ar1_log, fc_arx_log, fc_vix_log)),
    actual = actual_row$usdzar)
}

forecasts_long <- bind_rows(results) %>% mutate(error = actual - forecast)

rmse_tbl <- forecasts_long %>% group_by(model) %>%
  summarise(n = n(), rmse = sqrt(mean(error^2)), mae = mean(abs(error)),
            bias = mean(error), .groups = "drop") %>% arrange(rmse)

# 5. diebold-mariano test (testing RSME gap)
# ============================================================================
# H0: the two forecasts have equal predictive accuracy (squared-error loss)
# Two-sided; small-sample HLN correction (Harvey-Leybourne-Newbold)
paired <- forecasts_long %>%
  select(target_date, model, error) %>%
  pivot_wider(names_from = model, values_from = error) %>%
  arrange(target_date) %>% drop_na()

dm_pairs <- list( c("ar1", "arimax"), c("ar1", "arimax_vix"), c("arimax", "arimax_vix"))
for (p in dm_pairs) {
  dm <- dm.test(paired[[p[1]]], paired[[p[2]]],
                alternative = "two.sided", h = H, power = 2)
  cat(sprintf("DM %s vs %s: stat = %+.3f, p = %.3f\n",
              p[1], p[2], dm$statistic, dm$p.value))
}

# 6. bootstrap CI on the RMSE gap
# ============================================================================
B <- 10000
block_len <- H # 4 week block matches the forecast horizon
n <- nrow(paired)
starts_grid <- seq_len(n - block_len + 1)
n_blocks <- ceiling(n / block_len)

gap_star <- numeric(B)
for (b in seq_len(B)) {
  starts <- sample(starts_grid, n_blocks, replace = TRUE)
  idx <- unlist(lapply(starts, function(s) s:(s + block_len - 1)))
  idx <- idx[seq_len(n)]
  e_ar1 <- paired$ar1[idx]
  e_arx <- paired$arimax_vix[idx]
  gap_star[b] <- sqrt(mean(e_ar1^2)) - sqrt(mean(e_arx^2))   # RMSE(AR1) - RMSE(ARIMAX+VIX): >0 = our model wins
}

ci <- quantile(gap_star, c(0.025, 0.5, 0.975))
point_gap <- rmse_tbl$rmse[rmse_tbl$model == "ar1"] - rmse_tbl$rmse[rmse_tbl$model == "arimax_vix"]
cat(sprintf( "\n--- Bootstrap 95%% CI on RMSE(AR1) - RMSE(ARIMAX+VIX) ---\n
   Point estimate : %+ .4f Rand
   Bootstrap median: %+ .4f Rand
   95%% CI : [%+ .4f, %+ .4f] Rand
   Prob(ARIMAX wins on a resample): %.1f%%\n",
  point_gap, ci[2], ci[1], ci[3], 100 * mean(gap_star > 0)))

# Save numerics for the slide
dm_res <- dm.test(paired$ar1, paired$arimax_vix, alternative = "two.sided", h = H, power = 2)
bootstrap_out <- tibble(point_gap = point_gap, ci_lower = ci[1], ci_median = ci[2], ci_upper = ci[3],
  p_arimax_wins = mean(gap_star > 0),
  dm_stat = as.numeric(dm_res$statistic),
  dm_pvalue = as.numeric(dm_res$p.value))
write_csv(bootstrap_out, file.path(OUT_DIR, "significance_tests.csv"))

# Bootstrap distribution plot
p_boot <- tibble(gap = gap_star) %>%
  ggplot(aes(gap)) +
  geom_histogram(bins = 60, fill = "steelblue", alpha = 0.85) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = ci[c(1,3)], colour = "red", linetype = "dotted") +
  labs(title = "Bootstrap distribution: RMSE(AR1) - RMSE(ARIMAX+VIX)",
       subtitle = sprintf("B = %d, block length = %d weeks; red = 95%% CI",
                          B, block_len),
       x = "RMSE gap (Rand); >0 means ARIMAX+VIX wins", y = "count") +
  theme_minimal(base_size = 12)
ggsave(file.path(OUT_DIR, "plot_bootstrap_rmse_gap.png"), p_boot, width = 8, height = 4.5, dpi = 200)

# Diagnostics
# a. VIF — check for predictor redundancy
# ----------------------------------------------------------------------------

vif_manual <- function(X) {
  sapply(seq_len(ncol(X)), function(j) {
    r2 <- summary(lm(X[, j] ~ X[, -j, drop = FALSE]))$r.squared
    1 / (1 - r2)
  }) |> setNames(colnames(X))
}
X_all <- as.matrix(panel_use[, vars4])
vif_tbl <- tibble(predictor = vars4, vif = round(vif_manual(X_all), 3))
print(vif_tbl)
write_csv(vif_tbl, file.path(OUT_DIR, "vif_table.csv"))

# b. VIX screening (for validitity check)
# ----------------------------------------------------------------------------
screen_df <- forecasts_long %>%
  filter(model == "ar1") %>%
  mutate(log_err = log(actual) - log(forecast)) %>%
  left_join(panel_use %>%
              transmute(forecast_date = date, log_vix = log(vix),
                        dlog_vix_0 = dlog_vix, # week ending AT origin
                        dlog_vix_lag = lag(dlog_vix, 1)), # lagged (info set safe)
            by = "forecast_date") %>%
  drop_na(log_vix, dlog_vix_0, dlog_vix_lag)

for (v in c("log_vix", "dlog_vix_0", "dlog_vix_lag")) {
  m  <- lm(reformulate(v, "log_err"), data = screen_df)
  ct <- coeftest(m, vcov = NeweyWest(m, lag = 8))
  cat(sprintf("%-13s beta = %+.4f  t = %+.2f  p = %.3f  R2 = %.4f\n",
              v, ct[2,1], ct[2,3], ct[2,4], summary(m)$r.squared))
}
                       
# c. AIC/BIC ladder (checking for complexity) 
# ----------------------------------------------------------------------------
y_all <- log(panel_use$usdzar)
ladder <- list("AR(1) only" = NULL, "+ rate_diff" = "rate_diff_lag", "+ infl_diff" = "infl_diff_lag", "+ dlog_platinum" = "dlog_platinum_lag",
  "3-predictor ARIMAX"= vars3, "ARIMAX + dlog_vix" = vars4)

ic_tbl <- imap_dfr(ladder, function(v, nm) {
  xr  <- if (is.null(v)) NULL else as.matrix(panel_use[, v, drop = FALSE])
  fit <- Arima(y_all, order = c(1,0,0), xreg = xr)
  tibble(model = nm, k = length(coef(fit)), AIC = AIC(fit), BIC = BIC(fit))
}) %>% mutate(across(c(AIC, BIC), ~ round(., 1)))

write_csv(ic_tbl, file.path(OUT_DIR, "aic_bic_ladder.csv"))
# note: this is fitted on the full sample so it is an in-sample complexity - i shall use in the slides for presentation
# checking that complements never replaces the rolling out-of-sample RMSE which we got and it remains the headline evidence

# 7. future forecast: next 4 fridays from 24/07/2026
# ============================================================================
origin_future <- as.Date("2026-07-24")
train_fut <- panel_use %>% filter(date <= origin_future)
if (nrow(train_fut) == 0) {
  warning("No data through 2026-07-24 yet; using latest available Friday as origin.")
  origin_future <- max(panel_use$date)
  train_fut <- panel_use %>% filter(date <= origin_future)
}

y_fut <- log(train_fut$usdzar)

# regressor matrices: 3-predictor (original) and 4-predictor (+ dlog_vix)
x3_fut_tr <- as.matrix(train_fut[, vars3])
x4_fut_tr <- as.matrix(train_fut[, vars4])
x3_fut <- matrix(rep(tail(x3_fut_tr, 1), H), nrow = H, byrow = TRUE); colnames(x3_fut) <- vars3
x4_fut <- matrix(rep(tail(x4_fut_tr, 1), H), nrow = H, byrow = TRUE); colnames(x4_fut) <- vars4

fit_ar1_fut <- Arima(y_fut, order = c(1,0,0))
fit_arx_fut <- Arima(y_fut, order = c(1,0,0), xreg = x3_fut_tr)
fit_vix_fut <- Arima(y_fut, order = c(1,0,0), xreg = x4_fut_tr)

fc_ar1 <- forecast(fit_ar1_fut, h = H, level = 95)
fc_arx <- forecast(fit_arx_fut, h = H, xreg = x3_fut, level = 95)
fc_vix <- forecast(fit_vix_fut, h = H, xreg = x4_fut, level = 95)

future_dates <- origin_future + 7 * seq_len(H)
future_tbl <- tibble(
  target_date = future_dates,
  ar1_mean = as.numeric(exp(fc_ar1$mean)),
  ar1_lo = as.numeric(exp(fc_ar1$lower)),
  ar1_hi = as.numeric(exp(fc_ar1$upper)),
  arimax_mean = as.numeric(exp(fc_arx$mean)),
  arimax_lo = as.numeric(exp(fc_arx$lower)),
  arimax_hi = as.numeric(exp(fc_arx$upper)),
  arimax_vix_mean = as.numeric(exp(fc_vix$mean)),
  arimax_vix_lo = as.numeric(exp(fc_vix$lower)),
  arimax_vix_hi = as.numeric(exp(fc_vix$upper)))

write_csv(future_tbl, file.path(OUT_DIR, "future_forecast_4wk.csv"))

hist_tail <- panel_use %>%
  filter(date >= origin_future - 7 * 26 & date <= origin_future) %>%
  select(date, usdzar)

p_future <- ggplot() +
  geom_line(data = hist_tail, aes(date, usdzar), colour = "black", linewidth = 0.6) +
  geom_ribbon(data = future_tbl,
              aes(target_date, ymin = arimax_vix_lo, ymax = arimax_vix_hi),
              fill = "red", alpha = 0.15) +
  geom_line(data = future_tbl, aes(target_date, arimax_vix_mean, colour = "ARIMAX+VIX"),
            linewidth = 0.7) +
  geom_line(data = future_tbl, aes(target_date, ar1_mean, colour = "AR(1)"),
            linewidth = 0.7, linetype = "dashed") +
  geom_vline(xintercept = as.numeric(origin_future),
             linetype = "dotted", colour = "grey40") +
  scale_colour_manual(values = c("ARIMAX+VIX" = "red", "AR(1)" = "skyblue")) +
  labs(title = sprintf("USDZAR forecast: next 4 Fridays from %s",
                       format(origin_future, "%d %b %Y")),
       subtitle = "Black = last 26 weeks actual. Shaded band = ARIMAX+VIX 95% CI.",
       x = NULL, y = "Rand per USD", colour = "Model") +
  theme_minimal(base_size = 12) + theme(legend.position = "top")

ggsave(file.path(OUT_DIR, "plot_future_forecast.png"), p_future, width = 9, height = 4.8, dpi = 200)

# 8. plots
# ============================================================================
model_cols <- c(ar1 = "skyblue", arimax = "grey55", arimax_vix = "red")

# 1. forecasts vs actuals over the eval window
p_fc <- forecasts_long %>%
  ggplot(aes(x = target_date)) +
  geom_line(aes(y = actual), colour = "black", linewidth = 0.6) +
  geom_line(aes(y = forecast, colour = model), linewidth = 0.5, alpha = 0.9) +
  scale_colour_manual(values = model_cols) +
  labs(title = "USDZAR: 4 week ahead forecasts vs actual",
       subtitle = "Weekly friday cadence, expanding window, 2021–2025",
       x = NULL, y = "USDZAR", colour = "Model") +
  theme_minimal(base_size = 12)

# 2. RMSE bar chart with numeric labels (the rules asks for numbers on it)
p_rmse <- rmse_tbl %>%
  ggplot(aes(x = reorder(model, rmse), y = rmse, fill = model)) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.4f", rmse)),
            vjust = -0.4, size = 4.2) +
  scale_fill_manual(values = model_cols) +
  labs(title    = "Out-of-sample RMSE (Rand)",
       subtitle = sprintf("4 week ahead, %d friday origins, 2021–2025",
                          length(unique(forecasts_long$forecast_date))),
       x = NULL, y = "RMSE (Rand)") +
  expand_limits(y = max(rmse_tbl$rmse) * 1.15) +
  theme_minimal(base_size = 12)

# 3. Rolling 26 week RMSE (showing where each model wins/loses through time)
rolling_rmse <- forecasts_long %>%
  arrange(target_date) %>%
  group_by(model) %>%
  mutate(sq_err = error^2, roll_rmse = sqrt(zoo::rollmeanr(sq_err, k = 26, fill = NA))) %>%
  ungroup()

p_roll <- rolling_rmse %>%
  filter(!is.na(roll_rmse)) %>%
  ggplot(aes(x = target_date, y = roll_rmse, colour = model)) +
  geom_line(linewidth = 0.6) +
  scale_colour_manual(values = model_cols) +
  labs(title = "Rolling 26 week RMSE",
       subtitle = "Lower = better; shows regime-specific performance",
       x = NULL, y = "RMSE (Rand)", colour = "Model") +
  theme_minimal(base_size = 12)

# 9. exports
# ----------------------------------------------------------------
out_dir <- "outputs"
dir.create(out_dir, showWarnings = FALSE)

write_csv(forecasts_long, file.path(OUT_DIR, "forecasts_long.csv"))
write_csv(rmse_tbl, file.path(OUT_DIR, "rmse_summary.csv"))

ggsave(file.path(OUT_DIR, "plot_forecasts_vs_actual.png"), p_fc, width = 9, height = 4.5, dpi = 200)
ggsave(file.path(OUT_DIR, "plot_rmse_bar.png"), p_rmse, width = 6, height = 4.0, dpi = 200)
ggsave(file.path(OUT_DIR, "plot_rolling_rmse.png"), p_roll, width = 9, height = 4.5, dpi = 200)

cat("\nSaved to ./outputs/:\n", " forecasts_long.csv (schema: forecast_date, target_date, model, forecast, actual, error)\n",
    " rmse_summary.csv (model, n, rmse, mae, bias; 3 models)\n", " significance_tests.csv (bootstrap CI + DM test, AR1 vs ARIMAX+VIX)\n",
    " vif_table.csv / aic_bic_ladder.csv (predictor diagnostics)\n", " future_forecast_4wk.csv (all 3 models, 95% CIs)\n",
    " plot_forecasts_vs_actual.png\n", " plot_rmse_bar.png\n", " plot_rolling_rmse.png\n", " plot_bootstrap_rmse_gap.png\n",
    "  plot_future_forecast.png\n", sep = "")

# ================================================================
# plots

panel_series <- fx %>%
  left_join(repo_sa, by = "date") %>%
  left_join(fed_ffr, by = "date") %>%
  left_join(cpi_sa, by = "date") %>%
  left_join(cpi_us, by = "date") %>%
  left_join(platinum, by = "date") %>%
  left_join(vix_friday, by = "date") %>%
  arrange(date) %>%
  mutate(sa_repo = zoo::na.locf(sa_repo, na.rm = FALSE),
         us_ffr = zoo::na.locf(us_ffr, na.rm = FALSE),
         cpi_sa = zoo::na.locf(cpi_sa, na.rm = FALSE),
         cpi_us = zoo::na.locf(cpi_us, na.rm = FALSE),
         price = zoo::na.locf(price, na.rm = FALSE),
         vix = zoo::na.locf(vix, na.rm = FALSE),
         
         rate_diff = sa_repo - us_ffr,
         cpi_sa_yoy = log(cpi_sa) - log(lag(cpi_sa, 52)),
         cpi_us_yoy = log(cpi_us) - log(lag(cpi_us, 52)),
         infl_diff = cpi_sa_yoy - cpi_us_yoy
  ) %>%
  filter(if_all(c(rate_diff, infl_diff, price, vix), ~ !is.na(.)))

panel_plot <- panel_series %>% filter(date >= as.Date("2009-01-01"))

theme_series <- theme_minimal(base_size = 11) +
  theme(legend.position = "top", plot.title = element_text(face = "bold", size = 12),
        panel.grid.minor = element_blank())

p1 <- ggplot(panel_plot, aes(date, usdzar)) +
  geom_line(colour = "black", linewidth = 0.5) +
  labs(title = "USDZAR (weekly, Friday)", x = NULL, y = "Rand per USD") +
  theme_series

p2 <- ggplot(panel_plot) +
  geom_line(aes(date, sa_repo, colour = "SA Repo"), linewidth = 0.5) +
  geom_line(aes(date, us_ffr,  colour = "US Fed Funds"), linewidth = 0.5) +
  scale_colour_manual(values = c("SA Repo" = "red", "US Fed Funds" = "skyblue")) +
  labs(title = "Policy rates -> rate_diff = SA Repo - US Fed Funds",
       x = NULL, y = "%", colour = NULL) +
  theme_series

p3 <- ggplot(panel_plot, aes(date, infl_diff)) +
  geom_line(colour = "darkorange", linewidth = 0.5) +
  geom_hline(yintercept = 0, colour = "grey50", linewidth = 0.3) +
  labs(title = "infl_diff = SA CPI YoY - US CPI YoY (52-week log change)",
       x = NULL, y = "YoY differential") +
  theme_series

p4 <- ggplot(panel_plot, aes(date, price)) +
  geom_line(colour = "seagreen", linewidth = 0.5) +
  labs(title = "Platinum futures price (weekly, level used for dlog_platinum)",
       x = NULL, y = "USD / oz") +
  theme_series

p5 <- ggplot(panel_plot, aes(date, vix)) +
  geom_line(colour = "purple", linewidth = 0.5) +
  labs(title = "VIX index (weekly Friday close, level used for dlog_vix)",
       x = NULL, y = "Index level") +
  theme_series

combined <- (p1 / p2 / p3 / p4 / p5) +
  plot_annotation(title = "Series used in the model",
                  subtitle = sprintf("%s to %s -- the actual window the modelling panel uses",
                                     format(min(panel_plot$date), "%b %Y"),
                                     format(max(panel_plot$date), "%b %Y")),
                  theme = theme(plot.title = element_text(face = "bold", size = 14)))

ggsave(file.path(OUT_DIR, "plot_input_series.png"), combined, width = 9, height = 13.5, dpi = 200)
cat("Saved outputs/plot_input_series.png\n")
cat(sprintf("Plotted window: %s to %s (%d weekly obs)\n",
            min(panel_plot$date), max(panel_plot$date), nrow(panel_plot)))
