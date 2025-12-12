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


DEFAULT_TREND_SIGNAL_CONFIG = TrendContinuationSignalConfig(
    fast_ma_period=20,
    slow_ma_period=50,
    atr_period=14,
    atr_percentile=50,
    h4_sma_period=50,
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


SignalConfigType = Union[SignalConfig, MeanReversionSignalConfig, TrendContinuationSignalConfig]


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


@dataclass(frozen=True)
class PortfolioProfile:
    name: str
    strategies: List[StrategyProfile]
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
