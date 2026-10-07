# References

Every externally sourced figure in the code, README and deck traces to one of these. Figures for the
April 2025 tariff shock are measured directly from `data/prices`.

## Regulation and accounting

- Basel Committee on Banking Supervision, *Basel Framework, CRE20: Standardised approach - individual
  exposures*. Table 1 (sovereign risk weights 0/20/50/100/150%), Table 10 (corporate risk weights
  20/50/75/100/150%), CRE20.57 (equity 250%), CRE20.98-99 (credit conversion factors 40% for commitments,
  20% for short-term trade letters of credit).
  https://www.bis.org/committees/bcbs/basel-framework/81421/chapter-no-footnotes.pdf
- IFRS 9 *Financial Instruments*, expected credit loss model: 12-month ECL in Stage 1, lifetime ECL after a
  significant increase in credit risk (Stage 2). RiskPulse uses "stressed PD at least 2x base PD" as a
  simplified Stage 2 trigger.

## Stress-scenario analogues (config/shocks.yaml)

| Episode | Figures used | Source |
|---|---|---|
| US tariff shock, 2-8 Apr 2025 | SPY -12.1%, WTI -16.9%, VIX 21 -> 52, UST 10y +29bp to 11 Apr (+51bp off the 4 Apr low) | `data/prices/prices_daily.csv` |
| same | US high-yield spreads ~+110bp over 2-8 Apr; S&P 500 ~-11.5% | NEPC, *April 2025 market commentary*, https://www.nepc.com/nepcs-april-2025-market-commentary ; SIFMA Insights, *Monthly Metrics April 2025* |
| COVID crash, Feb-Mar 2020 | S&P 500 ~-35% peak to trough; US high yield 357 -> 1,087bp; BBB spread ~1.67% -> 4.80% | Columbia Threadneedle fixed-income performance update (2020); ICI research note on the March 2020 bond market |
| Russia-Ukraine, Feb-Mar 2022 | S&P 500 ~-12% YTD by the invasion, ~-7% in the prior week; Brent above $100 | The Boston Globe, 24 Feb 2022; Morningstar, 24 Feb 2022 |
| Regional-bank stress, Mar 2023 | KBW Nasdaq Bank index -15.8% in one week; 2-year UST -~100bp in three days, the largest since Oct 1987 | CNBC, 13 Mar 2023; Reuters via Kitco, 13 Mar 2023 |
| 2022 Fed tightening | UST 10y ~1.5% -> ~3.9% over 2022 | public Treasury yield history |

Magnitudes are approximate and used to give each scenario a plausible size. They are not a regulatory
scenario.

## Research

- Tetlock, P. (2007). Giving Content to Investor Sentiment: The Role of Media in the Stock Market.
  *Journal of Finance* 62(3), 1139-1168. High media pessimism predicts temporary downward price pressure
  that later reverts, which matches RiskPulse's finding that day-stamped news has no lasting next-day edge.
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., Takala, P. (2014). Good debt or bad debt: Detecting
  semantic orientations in economic texts. *JASIST* 65(4). (Financial PhraseBank.)
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
  arXiv:1908.10063. Model: https://huggingface.co/ProsusAI/finbert
- Xiao, S. et al. (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE models).
  arXiv:2309.07597. Model: https://huggingface.co/BAAI/bge-small-en-v1.5
- Hutto, C., Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social
  Media Text. *ICWSM*.

## Datasets

See `data/README.md` for every dataset, its licence and the script that retrieves it.
