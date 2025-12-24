from dataclasses import dataclass
from typing import Sequence, Union, List


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    pip_size: float  # minimum price increment
    pip_value_per_lot: float  # account currency value per pip for 1.0 lot


XAUUSD_SPEC = InstrumentSpec(
    symbol="XAUUSD",
    pip_size=0.01,
    pip_value_per_lot=1.0,  # 1 USD per 0.01 move per lot
)


@dataclass(frozen=True)
class TradingCosts:
    spread_pips: float  # one-way spread measured in pips
    commission_per_lot_round_trip: float  # currency per 1.0 lot round-trip


DEFAULT_COSTS = TradingCosts(
    spread_pips=20.0,  # 0.20 in price with pip_size=0.01
    commission_per_lot_round_trip=7.0,
)


@dataclass(frozen=True)
class ChallengeProfile:
    name: str
    profit_target_pct: float
    max_total_loss_pct: float
    min_bars_per_eval: int
    daily_loss_pct: float | None = None


DEFAULT_CHALLENGE = ChallengeProfile(
    name="Simple_7_6",
    profit_target_pct=0.07,
    max_total_loss_pct=0.06,
    min_bars_per_eval=500,
    daily_loss_pct=0.02,
)


@dataclass(frozen=True)
class TradingSession:
    name: str
    allowed_weekdays: Sequence[int]  # 0=Mon ... 6=Sun
    start_hour: int  # inclusive, 0–23
    end_hour: int  # exclusive, 1–24


DEFAULT_SESSION = TradingSession(
    name="XAU_London_NY",
    allowed_weekdays=(0, 1, 2, 3, 4),  # Mon–Fri
    start_hour=7,
    end_hour=20,
)


@dataclass(frozen=True)
class SignalConfig:
    breakout_lookback: int = 20
    atr_period: int = 14
    atr_percentile: int = 50
    h4_sma_period: int = 50


DEFAULT_SIGNAL_CONFIG = SignalConfig(
    breakout_lookback=20,
    atr_period=14,
    atr_percentile=50,
    h4_sma_period=50,
)


@dataclass(frozen=True)
class MeanReversionSignalConfig:
    ma_period: int = 50
    atr_period: int = 14
    entry_k: float = 1.0
    exit_k: float = 0.0
    h4_sma_period: int = 50


DEFAULT_MR_SIGNAL_CONFIG = MeanReversionSignalConfig(
    ma_period=50,
    atr_period=14,
    entry_k=1.5,
    exit_k=0.0,
    h4_sma_period=50,
)


@dataclass(frozen=True)
class TrendContinuationSignalConfig:
    fast_ma_period: int = 20
    slow_ma_period: int = 50
    atr_period: int = 14
    atr_percentile: int = 50
    h4_sma_period: int = 50


@dataclass(frozen=True)
class LondonBreakoutSignalConfig:
    pre_session_start_hour: int = 0
    pre_session_end_hour: int = 7
    breakout_buffer_pips: float = 0.0
    atr_period: int = 14
    atr_filter_percentile: int = 50
    atr_min_percentile: int = 40
    trend_tf: str = "H1"
    trend_strength_min: float = 0.0
    max_trades_per_day: int = 2


@dataclass(frozen=True)
class LiquiditySweepSignalConfig:
    lookback_levels: int = 50
    sweep_threshold_pips: float = 10.0
    fvg_depth: int = 3
    body_reentry_required: bool = True
    atr_period: int = 14
    atr_filter_percentile: int = 50
    trend_tf: str = "H4"
    trend_strength_min: float = 0.0
    max_trades_per_day: int = 2


@dataclass(frozen=True)
class MomentumPinballSignalConfig:
    trend_ma_period: int = 20
    rsi_period: int = 2
    rsi_oversold: int = 10
    rsi_overbought: int = 90
    atr_period: int = 14
    atr_filter_percentile: int = 75
    max_trades_per_day: int = 2
    ny_session_start_hour: int = 13
    ny_session_end_hour: int = 17
    atr_min_percentile: int = 70


@dataclass(frozen=True)
class TrendKDSignalConfig:
    atr_percentile: int = 70
    m15_ema_period: int = 20
    h1_ema_period: int = 20
    h1_slope_threshold: float = 0.0001
    max_trades_per_day: int = 1


@dataclass(frozen=True)
class RiskControlConfig:
    dd_reduce_threshold_pct: float = 0.05
    dd_restore_threshold_pct: float = 0.03
    risk_scale_reduction_factor: float = 0.5


@dataclass(frozen=True)
class VanVleetSignalConfig:
    fast_ma_period: int = 10
    slow_ma_period: int = 30
    pullback_ma_period: int = 20
    atr_period: int = 14
    atr_percentile: int = 50
    rsi_period: int = 14
    rsi_pullback: int = 40
    max_trades_per_day: int | None = None


@dataclass(frozen=True)
class BigManSignalConfig:
    sma_period: int = 50
    adx_period: int = 14
    adx_max: float = 25.0
    atr_period: int = 14
    atr_percentile: int = 40
    band_k: float = 1.0
    max_trades_per_day: int | None = None


DEFAULT_TREND_SIGNAL_CONFIG = TrendContinuationSignalConfig(
    fast_ma_period=20,
    slow_ma_period=50,
    atr_period=14,
    atr_percentile=50,
    h4_sma_period=50,
)

DEFAULT_LONDON_BREAKOUT_SIGNAL_CONFIG = LondonBreakoutSignalConfig(
    pre_session_start_hour=0,
    pre_session_end_hour=7,
    breakout_buffer_pips=0.0,
    atr_period=14,
    atr_filter_percentile=50,
)

DEFAULT_LIQUIDITY_SWEEP_SIGNAL_CONFIG = LiquiditySweepSignalConfig(
    lookback_levels=50,
    sweep_threshold_pips=10.0,
    fvg_depth=3,
    body_reentry_required=True,
    atr_period=14,
    atr_filter_percentile=50,
)

# GBPJPY-tuned configs (research)
DEFAULT_LONDON_BREAKOUT_SIGNAL_CONFIG_GBPJPY = LondonBreakoutSignalConfig(
    pre_session_start_hour=0,
    pre_session_end_hour=7,
    breakout_buffer_pips=5.0,
    atr_period=14,
    atr_filter_percentile=30,
    atr_min_percentile=30,
)

DEFAULT_LIQUIDITY_SWEEP_SIGNAL_CONFIG_GBPJPY = LiquiditySweepSignalConfig(
    lookback_levels=60,
    sweep_threshold_pips=3.0,
    fvg_depth=3,
    body_reentry_required=True,
    atr_period=14,
    atr_filter_percentile=70,
)

DEFAULT_MOMENTUM_PINBALL_CONFIG_M5 = MomentumPinballSignalConfig(
    trend_ma_period=20,
    rsi_period=2,
    rsi_oversold=10,
    rsi_overbought=90,
    atr_period=14,
    atr_filter_percentile=75,
    max_trades_per_day=2,
    atr_min_percentile=70,
)

DEFAULT_TREND_KD_SIGNAL_CONFIG = TrendKDSignalConfig(
    atr_percentile=70,
    m15_ema_period=20,
    h1_ema_period=20,
    h1_slope_threshold=0.0001,
    max_trades_per_day=1,
)

DEFAULT_VANVLEET_SIGNAL_CONFIG = VanVleetSignalConfig()
DEFAULT_BIGMAN_SIGNAL_CONFIG = BigManSignalConfig()

# Tighter research configs for EXP_V3 (regime + caps)
EXP_V3_LONDON_SIGNAL_CONFIG = LondonBreakoutSignalConfig(
    pre_session_start_hour=0,
    pre_session_end_hour=7,
    breakout_buffer_pips=0.0,
    atr_period=14,
    atr_filter_percentile=50,
    atr_min_percentile=40,
    trend_tf="H1",
    trend_strength_min=0.0,
    max_trades_per_day=1,
)

EXP_V3_LIQUI_SIGNAL_CONFIG = LiquiditySweepSignalConfig(
    lookback_levels=50,
    sweep_threshold_pips=10.0,
    fvg_depth=3,
    body_reentry_required=True,
    atr_period=14,
    atr_filter_percentile=50,
    trend_tf="H4",
    trend_strength_min=0.0,
    max_trades_per_day=1,
)

EXP_V3_KD_SIGNAL_CONFIG = TrendKDSignalConfig(
    atr_percentile=70,
    m15_ema_period=20,
    h1_ema_period=20,
    h1_slope_threshold=0.0001,
    max_trades_per_day=2,
)

EXP_V3_VANVLEET_SIGNAL_CONFIG = VanVleetSignalConfig(
    fast_ma_period=10,
    slow_ma_period=30,
    pullback_ma_period=20,
    atr_period=14,
    atr_percentile=50,
    rsi_period=14,
    rsi_pullback=40,
    max_trades_per_day=2,
)

EXP_V3_BIGMAN_SIGNAL_CONFIG = BigManSignalConfig(
    sma_period=50,
    adx_period=14,
    adx_max=25.0,
    atr_period=14,
    atr_percentile=40,
    band_k=1.0,
    max_trades_per_day=2,
)

@dataclass(frozen=True)
class StrategyConfig:
    symbol: str
    fixed_lot_size: float
    profit_target_pct: float  # e.g. 0.07 for 7% account target
    risk_per_trade_pct: float  # 0.02 = 2% of account
    reward_per_trade_pct: float  # 0.04 = 4% of account


DEFAULT_STRATEGY = StrategyConfig(
    symbol="XAUUSD",  # gold
    fixed_lot_size=2.0,
    profit_target_pct=0.07,  # 7% account target
    risk_per_trade_pct=0.02,  # 2% risk
    reward_per_trade_pct=0.04,  # 4% reward
)


SignalConfigType = Union[
    SignalConfig,
    MeanReversionSignalConfig,
    TrendContinuationSignalConfig,
    LondonBreakoutSignalConfig,
    LiquiditySweepSignalConfig,
    MomentumPinballSignalConfig,
    "TrendKDSignalConfig",
    "VanVleetSignalConfig",
    "BigManSignalConfig",
]


@dataclass(frozen=True)
class EdgeRegimeConfig:
    allow_in_low_vol: bool = False
    allow_in_high_vol_trend: bool = True
    allow_in_high_vol_reversal: bool = True
    allow_in_chop: bool = False


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    symbol_key: str
    timeframe: str
    strategy: StrategyConfig
    signals: SignalConfigType
    challenge: ChallengeProfile
    costs: TradingCosts
    session: TradingSession
    edge_regime_config: EdgeRegimeConfig | None = None


@dataclass(frozen=True)
class PortfolioProfile:
    name: str
    strategies: List[StrategyProfile]
    risk_scales: List[float] | None = None
    portfolio_daily_loss_pct: float | None = None
    portfolio_max_loss_pct: float | None = None
    risk_control: "RiskControlConfig | None" = None


@dataclass(frozen=True)
class MultiSymbolPortfolioProfile:
    name: str
    symbols: List[str]
    portfolios: List[PortfolioProfile]
    symbol_risk_scales: List[float] | None = None
    portfolio_daily_loss_pct: float | None = None
    portfolio_max_loss_pct: float | None = None


DEFAULT_PROFILE_BREAKOUT_V1 = StrategyProfile(
    name="XAU_H1_Breakout_V1",
    symbol_key="XAUUSD",
    timeframe="1h",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
)

DEFAULT_PROFILE_XAU_MR_V1 = StrategyProfile(
    name="XAU_H1_MeanReversion_V1",
    symbol_key="XAUUSD",
    timeframe="1h",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_MR_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
)

DEFAULT_PROFILE_USDJPY_MR_V1 = StrategyProfile(
    name="USDJPY_H1_MeanReversion_V1",
    symbol_key="USDJPY",
    timeframe="1h",
    strategy=DEFAULT_STRATEGY,
    signals=MeanReversionSignalConfig(
        ma_period=50,
        atr_period=14,
        entry_k=1.0,
        exit_k=0.0,
        h4_sma_period=50,
    ),
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
)

DEFAULT_PROFILE_USDJPY_MR_M15_V1 = StrategyProfile(
    name="USDJPY_M15_MeanReversion_V1",
    symbol_key="USDJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=MeanReversionSignalConfig(
        ma_period=50,
        atr_period=14,
        entry_k=1.0,
        exit_k=0.0,
        h4_sma_period=50,
    ),
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
)

DEFAULT_PROFILE_EURUSD_MR_M15_V1 = StrategyProfile(
    name="EURUSD_M15_MeanReversion_V1",
    symbol_key="EURUSD",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=MeanReversionSignalConfig(
        ma_period=50,
        atr_period=14,
        entry_k=1.0,
        exit_k=0.0,
        h4_sma_period=50,
    ),
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
)

DEFAULT_PROFILE_GBPUSD_MR_M15_V1 = StrategyProfile(
    name="GBPUSD_M15_MeanReversion_V1",
    symbol_key="GBPUSD",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=MeanReversionSignalConfig(
        ma_period=50,
        atr_period=14,
        entry_k=1.0,
        exit_k=0.0,
        h4_sma_period=50,
    ),
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
)

DEFAULT_PROFILE_XAUUSD_MR_M15_V1 = StrategyProfile(
    name="XAUUSD_M15_MeanReversion_V1",
    symbol_key="XAUUSD",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=MeanReversionSignalConfig(
        ma_period=50,
        atr_period=14,
        entry_k=1.0,
        exit_k=0.0,
        h4_sma_period=50,
    ),
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
)

DEFAULT_PROFILE_USDJPY_TREND_M15_V1 = StrategyProfile(
    name="USDJPY_M15_TrendCont_V1",
    symbol_key="USDJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_TREND_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
)

DEFAULT_PROFILE_USDJPY_LONDON_M15_V1 = StrategyProfile(
    name="USDJPY_M15_LondonBreakout_V1",
    symbol_key="USDJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_LONDON_BREAKOUT_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=False,
        allow_in_high_vol_trend=True,
        allow_in_high_vol_reversal=True,
        allow_in_chop=False,
    ),
)

DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1 = StrategyProfile(
    name="USDJPY_M15_LiquiditySweep_V1",
    symbol_key="USDJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_LIQUIDITY_SWEEP_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=False,
        allow_in_high_vol_trend=True,
        allow_in_high_vol_reversal=True,
        allow_in_chop=False,
    ),
)

DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1 = StrategyProfile(
    name="USDJPY_M5_MomentumPinball_V1",
    symbol_key="USDJPY",
    timeframe="5m",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_MOMENTUM_PINBALL_CONFIG_M5,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=False,
        allow_in_high_vol_trend=True,
        allow_in_high_vol_reversal=False,
        allow_in_chop=False,
    ),
)

EXPERIMENTAL_PROFILE_USDJPY_TREND_KD_M15_V1 = StrategyProfile(
    name="USDJPY_M15_TrendKD_V1",
    symbol_key="USDJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_TREND_KD_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=False,
        allow_in_high_vol_trend=True,
        allow_in_high_vol_reversal=False,
        allow_in_chop=False,
    ),
)

# EXP_V3-specific profiles (tighter regimes / caps) - research-only
EXP_V3_PROFILE_USDJPY_LONDON_M15 = StrategyProfile(
    name="USDJPY_M15_LondonBreakout_V3",
    symbol_key="USDJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=EXP_V3_LONDON_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=False,
        allow_in_high_vol_trend=True,
        allow_in_high_vol_reversal=False,
        allow_in_chop=False,
    ),
)

EXP_V3_PROFILE_USDJPY_LIQUI_M15 = StrategyProfile(
    name="USDJPY_M15_LiquiditySweep_V3",
    symbol_key="USDJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=EXP_V3_LIQUI_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=False,
        allow_in_high_vol_trend=False,
        allow_in_high_vol_reversal=True,
        allow_in_chop=False,
    ),
)

EXP_V3_PROFILE_USDJPY_MOMENTUM_M5 = StrategyProfile(
    name="USDJPY_M5_MomentumPinball_V1",
    symbol_key="USDJPY",
    timeframe="5m",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_MOMENTUM_PINBALL_CONFIG_M5,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=False,
        allow_in_high_vol_trend=True,
        allow_in_high_vol_reversal=False,
        allow_in_chop=False,
    ),
)

EXP_V3_PROFILE_USDJPY_TREND_KD_M15 = StrategyProfile(
    name="USDJPY_M15_TrendKD_V3",
    symbol_key="USDJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=EXP_V3_KD_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=False,
        allow_in_high_vol_trend=True,
        allow_in_high_vol_reversal=False,
        allow_in_chop=False,
    ),
)

EXP_V3_PROFILE_USDJPY_VANVLEET_M15 = StrategyProfile(
    name="USDJPY_M15_VanVleet_V1",
    symbol_key="USDJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=EXP_V3_VANVLEET_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=False,
        allow_in_high_vol_trend=True,
        allow_in_high_vol_reversal=False,
        allow_in_chop=False,
    ),
)

EXP_V3_PROFILE_USDJPY_BIGMAN_H1 = StrategyProfile(
    name="USDJPY_H1_BigMan_V1",
    symbol_key="USDJPY",
    timeframe="1h",
    strategy=DEFAULT_STRATEGY,
    signals=EXP_V3_BIGMAN_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=True,
        allow_in_high_vol_trend=False,
        allow_in_high_vol_reversal=False,
        allow_in_chop=False,
    ),
)

DEFAULT_PORTFOLIO_USDJPY_MR = PortfolioProfile(
    name="USDJPY_MR_Portfolio_V1",
    strategies=[
        DEFAULT_PROFILE_USDJPY_MR_V1,
        DEFAULT_PROFILE_USDJPY_MR_M15_V1,
    ],
    portfolio_daily_loss_pct=0.03,
    portfolio_max_loss_pct=0.06,
)

DEFAULT_PORTFOLIO_USDJPY_V2 = PortfolioProfile(
    name="USDJPY_Portfolio_V2",
    strategies=[
        DEFAULT_PROFILE_USDJPY_MR_M15_V1,
        DEFAULT_PROFILE_USDJPY_TREND_M15_V1,
    ],
    portfolio_daily_loss_pct=0.03,
    portfolio_max_loss_pct=0.06,
)

DEFAULT_PORTFOLIO_M15_MR_MULTI = PortfolioProfile(
    name="M15_MR_MultiSymbol_V1",
    strategies=[
        DEFAULT_PROFILE_USDJPY_MR_M15_V1,
        DEFAULT_PROFILE_EURUSD_MR_M15_V1,
        DEFAULT_PROFILE_GBPUSD_MR_M15_V1,
        DEFAULT_PROFILE_XAUUSD_MR_M15_V1,
    ],
    portfolio_daily_loss_pct=0.03,
    portfolio_max_loss_pct=0.06,
)

DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3 = PortfolioProfile(
    name="USDJPY_FastPass_V3",
    strategies=[
        DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
        DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
        DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
    ],
    risk_scales=[1.60, 0.25, 0.01],
    portfolio_daily_loss_pct=0.03,
    portfolio_max_loss_pct=0.095,
)

# Placeholder GBPJPY fast-pass edges (tuning later)
DEFAULT_PROFILE_GBPJPY_LONDON_M15_V1 = StrategyProfile(
    name="GBPJPY_M15_LondonBreakout_V1",
    symbol_key="GBPJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_LONDON_BREAKOUT_SIGNAL_CONFIG_GBPJPY,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=False,
        allow_in_high_vol_trend=True,
        allow_in_high_vol_reversal=True,
        allow_in_chop=False,
    ),
)

DEFAULT_PROFILE_GBPJPY_LIQUI_M15_V1 = StrategyProfile(
    name="GBPJPY_M15_LiquiditySweep_V1",
    symbol_key="GBPJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_LIQUIDITY_SWEEP_SIGNAL_CONFIG_GBPJPY,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
    edge_regime_config=EdgeRegimeConfig(
        allow_in_low_vol=False,
        allow_in_high_vol_trend=True,
        allow_in_high_vol_reversal=True,
        allow_in_chop=False,
    ),
)

DEFAULT_PORTFOLIO_GBPJPY_FASTPASS_CORE = PortfolioProfile(
    name="GBPJPY_FastPass_Core",
    strategies=[
        DEFAULT_PROFILE_GBPJPY_LONDON_M15_V1,
        DEFAULT_PROFILE_GBPJPY_LIQUI_M15_V1,
    ],
    risk_scales=[1.5, 0.5],
    portfolio_daily_loss_pct=0.03,
    portfolio_max_loss_pct=0.095,
)

# Multi-symbol FastPass (USDJPY V3 + GBPJPY Core)
MULTI_PORTFOLIO_USDJPY_GBPJPY_FASTPASS = MultiSymbolPortfolioProfile(
    name="FastPass_USDJPY_GBPJPY_V1",
    symbols=["USDJPY", "GBPJPY"],
    portfolios=[
        DEFAULT_PORTFOLIO_USDJPY_FASTPASS_V3,
        DEFAULT_PORTFOLIO_GBPJPY_FASTPASS_CORE,
    ],
    symbol_risk_scales=[1.0, 0.75],
    portfolio_daily_loss_pct=0.05,
    portfolio_max_loss_pct=0.095,
)

# Experimental portfolio (research-only)
EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2 = PortfolioProfile(
    name="USDJPY_FastPass_Exp_V2",
    strategies=[
        DEFAULT_PROFILE_USDJPY_LONDON_M15_V1,
        DEFAULT_PROFILE_USDJPY_LIQUI_M15_V1,
        DEFAULT_PROFILE_USDJPY_MOMENTUM_M5_V1,
        EXPERIMENTAL_PROFILE_USDJPY_TREND_KD_M15_V1,
    ],
    risk_scales=[1.6, 0.25, 0.01, 0.3],
    portfolio_daily_loss_pct=0.05,
    portfolio_max_loss_pct=0.095,
    risk_control=RiskControlConfig(
        dd_reduce_threshold_pct=0.05,
        dd_restore_threshold_pct=0.03,
        risk_scale_reduction_factor=0.5,
    ),
)

# Experimental V3 (with VanVleet + BigMan) -- research-only
EXPERIMENTAL_PROFILE_USDJPY_VANVLEET_M15_V1 = StrategyProfile(
    name="USDJPY_M15_VanVleet_V1",
    symbol_key="USDJPY",
    timeframe="15m",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_VANVLEET_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
)

EXPERIMENTAL_PROFILE_USDJPY_BIGMAN_H1_V1 = StrategyProfile(
    name="USDJPY_H1_BigMan_V1",
    symbol_key="USDJPY",
    timeframe="1h",
    strategy=DEFAULT_STRATEGY,
    signals=DEFAULT_BIGMAN_SIGNAL_CONFIG,
    challenge=DEFAULT_CHALLENGE,
    costs=DEFAULT_COSTS,
    session=DEFAULT_SESSION,
)

EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V3 = PortfolioProfile(
    name="USDJPY_FastPass_Exp_V3",
    strategies=[
        EXP_V3_PROFILE_USDJPY_LONDON_M15,
        EXP_V3_PROFILE_USDJPY_LIQUI_M15,
        EXP_V3_PROFILE_USDJPY_MOMENTUM_M5,
        EXP_V3_PROFILE_USDJPY_TREND_KD_M15,
        EXP_V3_PROFILE_USDJPY_VANVLEET_M15,
        EXP_V3_PROFILE_USDJPY_BIGMAN_H1,
    ],
    risk_scales=[1.6, 0.25, 0.01, 0.3, 0.3, 0.3],
    portfolio_daily_loss_pct=0.05,
    portfolio_max_loss_pct=0.095,
    risk_control=RiskControlConfig(
        dd_reduce_threshold_pct=0.05,
        dd_restore_threshold_pct=0.03,
        risk_scale_reduction_factor=0.5,
    ),
)

# Experimental multi-symbol (research-only)
MULTI_PORTFOLIO_USDJPY_EXP_GBPJPY_CORE = MultiSymbolPortfolioProfile(
    name="FastPass_USDJPY_EXP_GBPJPY_V1",
    symbols=["USDJPY", "GBPJPY"],
    portfolios=[
        EXPERIMENTAL_PORTFOLIO_USDJPY_EXP_V2,
        DEFAULT_PORTFOLIO_GBPJPY_FASTPASS_CORE,
    ],
    symbol_risk_scales=[1.0, 0.5],
    portfolio_daily_loss_pct=0.05,
    portfolio_max_loss_pct=0.095,
)

# FTMO-style challenge profile (10% target, 10% max loss, 5% daily)
FTMO_CHALLENGE_USDJPY = ChallengeProfile(
    name="FTMO_USDJPY",
    profit_target_pct=0.10,
    max_total_loss_pct=0.095,
    min_bars_per_eval=500,
    daily_loss_pct=0.05,
)
