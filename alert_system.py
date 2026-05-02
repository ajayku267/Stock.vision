"""
Automated alert system for stock price movements and technical indicators.
"""
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import redis
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import yfinance as yf

logger = logging.getLogger("stockvision.alerts")

Base = declarative_base()

class AlertType(Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PERCENTAGE_CHANGE = "percentage_change"
    VOLUME_SPIKE = "volume_spike"
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    SMA_CROSSOVER = "sma_crossover"
    PORTFOLIO_CHANGE = "portfolio_change"

class AlertStatus(Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    DISABLED = "disabled"
    EXPIRED = "expired"

@dataclass
class Alert:
    """Alert configuration."""
    id: Optional[int] = None
    user_id: int = 0
    ticker: str = ""
    alert_type: AlertType = AlertType.PRICE_ABOVE
    threshold: float = 0.0
    condition: str = ">"  # >, <, >=, <=, ==
    message_template: str = ""
    notification_channels: List[str] = field(default_factory=lambda: ["email"])
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    triggered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass
class AlertTrigger:
    """Alert trigger event."""
    alert_id: int
    ticker: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

# Database Models
class AlertModel(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    threshold = Column(Float, nullable=False)
    condition = Column(String(10), nullable=False)
    message_template = Column(Text)
    notification_channels = Column(Text)  # JSON string
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    triggered_at = Column(DateTime)
    expires_at = Column(DateTime)
    last_triggered = Column(DateTime)
    trigger_count = Column(Integer, default=0)

class AlertHistoryModel(Base):
    __tablename__ = "alert_history"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False)
    ticker = Column(String(10), nullable=False)
    current_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    message = Column(Text, nullable=False)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    notification_sent = Column(Boolean, default=False)

class NotificationChannel:
    """Base class for notification channels."""
    
    async def send_notification(self, alert: Alert, trigger: AlertTrigger) -> bool:
        """Send notification for alert trigger."""
        raise NotImplementedError

class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    async def send_notification(self, alert: Alert, trigger: AlertTrigger) -> bool:
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = f"user{alert.user_id}@example.com"  # Get from user DB
            msg['Subject'] = f"Stock Alert: {alert.ticker} - {alert.alert_type.value}"
            
            body = self._format_email_body(alert, trigger)
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _format_email_body(self, alert: Alert, trigger: AlertTrigger) -> str:
        """Format email body."""
        return f"""
        <html>
        <body>
            <h2>📈 Stock Alert Triggered</h2>
            <p><strong>Ticker:</strong> {alert.ticker}</p>
            <p><strong>Alert Type:</strong> {alert.alert_type.value}</p>
            <p><strong>Current Value:</strong> {trigger.current_value:.2f}</p>
            <p><strong>Threshold:</strong> {trigger.threshold:.2f}</p>
            <p><strong>Message:</strong> {trigger.message}</p>
            <p><strong>Time:</strong> {trigger.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <hr>
            <p><em>This is an automated message from StockVision Alert System</em></p>
        </body>
        </html>
        """

class WebSocketNotificationChannel(NotificationChannel):
    """WebSocket notification channel."""
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
    
    async def send_notification(self, alert: Alert, trigger: AlertTrigger) -> bool:
        """Send WebSocket notification."""
        try:
            message = {
                "type": "alert_triggered",
                "alert": {
                    "id": alert.id,
                    "ticker": alert.ticker,
                    "alert_type": alert.alert_type.value,
                    "threshold": alert.threshold
                },
                "trigger": {
                    "current_value": trigger.current_value,
                    "message": trigger.message,
                    "timestamp": trigger.timestamp.isoformat()
                }
            }
            
            await self.connection_manager.send_personal_message(message, f"user_{alert.user_id}")
            logger.info(f"WebSocket notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WebSocket notification: {e}")
            return False

class AlertEngine:
    """Main alert engine for monitoring and triggering alerts."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.active_alerts: Dict[int, Alert] = {}
        self.running = False
        
    def add_notification_channel(self, name: str, channel: NotificationChannel):
        """Add notification channel."""
        self.notification_channels[name] = channel
    
    async def add_alert(self, alert: Alert) -> int:
        """Add new alert."""
        # Save to database (mock implementation)
        alert.id = len(self.active_alerts) + 1
        self.active_alerts[alert.id] = alert
        
        # Cache in Redis
        alert_data = {
            "id": alert.id,
            "user_id": alert.user_id,
            "ticker": alert.ticker,
            "alert_type": alert.alert_type.value,
            "threshold": alert.threshold,
            "condition": alert.condition,
            "is_active": alert.is_active,
            "expires_at": alert.expires_at.isoformat() if alert.expires_at else None
        }
        
        self.redis_client.hset(f"alert:{alert.id}", mapping=alert_data)
        self.redis_client.sadd(f"alerts:ticker:{alert.ticker}", alert.id)
        
        logger.info(f"Added alert {alert.id} for {alert.ticker}")
        return alert.id
    
    async def remove_alert(self, alert_id: int) -> bool:
        """Remove alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        
        # Remove from cache
        self.redis_client.delete(f"alert:{alert_id}")
        self.redis_client.srem(f"alerts:ticker:{alert.ticker}", alert_id)
        
        # Remove from memory
        del self.active_alerts[alert_id]
        
        logger.info(f"Removed alert {alert_id}")
        return True
    
    async def check_alerts(self, ticker: str, current_data: Dict[str, float]) -> List[AlertTrigger]:
        """Check alerts for a ticker and return triggered alerts."""
        triggered_alerts = []
        
        # Get alerts for this ticker
        alert_ids = self.redis_client.smembers(f"alerts:ticker:{ticker}")
        
        for alert_id in alert_ids:
            alert_id = int(alert_id)
            if alert_id not in self.active_alerts:
                continue
            
            alert = self.active_alerts[alert_id]
            
            # Check if alert is active and not expired
            if not alert.is_active:
                continue
            
            if alert.expires_at and datetime.utcnow() > alert.expires_at:
                alert.is_active = False
                continue
            
            # Check alert condition
            if await self._evaluate_alert_condition(alert, current_data):
                trigger = AlertTrigger(
                    alert_id=alert.id,
                    ticker=ticker,
                    current_value=current_data.get('price', 0),
                    threshold=alert.threshold,
                    message=self._generate_alert_message(alert, current_data)
                )
                
                triggered_alerts.append(trigger)
                
                # Update alert
                alert.triggered_at = trigger.timestamp
                alert.last_triggered = trigger.timestamp
                alert.trigger_count += 1
                
                # Send notifications
                await self._send_notifications(alert, trigger)
        
        return triggered_alerts
    
    async def _evaluate_alert_condition(self, alert: Alert, current_data: Dict[str, float]) -> bool:
        """Evaluate if alert condition is met."""
        current_value = self._get_value_for_alert_type(alert, current_data)
        
        if current_value is None:
            return False
        
        condition = alert.condition
        threshold = alert.threshold
        
        if condition == ">":
            return current_value > threshold
        elif condition == "<":
            return current_value < threshold
        elif condition == ">=":
            return current_value >= threshold
        elif condition == "<=":
            return current_value <= threshold
        elif condition == "==":
            return abs(current_value - threshold) < 0.01  # Small tolerance
        
        return False
    
    def _get_value_for_alert_type(self, alert: Alert, current_data: Dict[str, float]) -> Optional[float]:
        """Get current value for alert type."""
        if alert.alert_type in [AlertType.PRICE_ABOVE, AlertType.PRICE_BELOW]:
            return current_data.get('price')
        elif alert.alert_type == AlertType.PERCENTAGE_CHANGE:
            return current_data.get('change_percent')
        elif alert.alert_type == AlertType.VOLUME_SPIKE:
            return current_data.get('volume')
        elif alert.alert_type in [AlertType.RSI_OVERSOLD, AlertType.RSI_OVERBOUGHT]:
            return current_data.get('rsi')
        elif alert.alert_type == AlertType.SMA_CROSSOVER:
            return current_data.get('sma_ratio')
        
        return None
    
    def _generate_alert_message(self, alert: Alert, current_data: Dict[str, float]) -> str:
        """Generate alert message."""
        current_value = self._get_value_for_alert_type(alert, current_data)
        
        if alert.message_template:
            return alert.message_template.format(
                ticker=alert.ticker,
                current_value=current_value,
                threshold=alert.threshold
            )
        
        # Default message templates
        templates = {
            AlertType.PRICE_ABOVE: f"{alert.ticker} price crossed above ${alert.threshold:.2f} (currently ${current_value:.2f})",
            AlertType.PRICE_BELOW: f"{alert.ticker} price fell below ${alert.threshold:.2f} (currently ${current_value:.2f})",
            AlertType.PERCENTAGE_CHANGE: f"{alert.ticker} changed by {current_value:.2f}% (threshold: {alert.threshold:.2f}%)",
            AlertType.VOLUME_SPIKE: f"{alert.ticker} volume spike detected: {current_value:,.0f} shares",
            AlertType.RSI_OVERSOLD: f"{alert.ticker} RSI oversold: {current_value:.2f}",
            AlertType.RSI_OVERBOUGHT: f"{alert.ticker} RSI overbought: {current_value:.2f}",
            AlertType.SMA_CROSSOVER: f"{alert.ticker} SMA crossover detected: ratio {current_value:.3f}"
        }
        
        return templates.get(alert.alert_type, f"{alert.ticker} alert triggered: {current_value}")
    
    async def _send_notifications(self, alert: Alert, trigger: AlertTrigger):
        """Send notifications through configured channels."""
        for channel_name in alert.notification_channels:
            if channel_name in self.notification_channels:
                try:
                    await self.notification_channels[channel_name].send_notification(alert, trigger)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel_name}: {e}")
    
    async def start_monitoring(self, tickers: List[str], check_interval: int = 60):
        """Start monitoring alerts for specified tickers."""
        self.running = True
        logger.info(f"Started alert monitoring for {len(tickers)} tickers")
        
        while self.running:
            try:
                # Get current data for all tickers
                for ticker in tickers:
                    try:
                        # Get current stock data
                        stock = yf.Ticker(ticker)
                        data = stock.history(period="1d", interval="1m")
                        
                        if not data.empty:
                            latest = data.iloc[-1]
                            
                            # Calculate technical indicators
                            current_data = {
                                'price': float(latest['Close']),
                                'change_percent': float(((latest['Close'] - latest['Open']) / latest['Open']) * 100),
                                'volume': int(latest['Volume']),
                                'rsi': self._calculate_rsi(data),
                                'sma_ratio': self._calculate_sma_ratio(data)
                            }
                            
                            # Check alerts
                            triggered = await self.check_alerts(ticker, current_data)
                            if triggered:
                                logger.info(f"Triggered {len(triggered)} alerts for {ticker}")
                    
                    except Exception as e:
                        logger.error(f"Error monitoring {ticker}: {e}")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI indicator."""
        try:
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not rsi.empty else 50.0
        except:
            return 50.0
    
    def _calculate_sma_ratio(self, data: pd.DataFrame, short_period: int = 20, long_period: int = 50) -> float:
        """Calculate SMA ratio (short/long)."""
        try:
            short_sma = data['Close'].rolling(window=short_period).mean()
            long_sma = data['Close'].rolling(window=long_period).mean()
            ratio = short_sma / long_sma
            return float(ratio.iloc[-1]) if not ratio.empty else 1.0
        except:
            return 1.0
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self.running = False
        logger.info("Stopped alert monitoring")

# Alert manager for easy access
class AlertManager:
    """High-level alert management interface."""
    
    def __init__(self, redis_client: redis.Redis, connection_manager=None):
        self.alert_engine = AlertEngine(redis_client)
        
        # Add default notification channels
        if connection_manager:
            self.alert_engine.add_notification_channel("websocket", WebSocketNotificationChannel(connection_manager))
        
        # Email channel would be added with actual SMTP credentials
        # self.alert_engine.add_notification_channel("email", EmailNotificationChannel(...))
    
    async def create_price_alert(self, user_id: int, ticker: str, price_threshold: float, 
                                 above: bool = True, notification_channels: List[str] = None) -> int:
        """Create price alert."""
        alert_type = AlertType.PRICE_ABOVE if above else AlertType.PRICE_BELOW
        condition = ">" if above else "<"
        
        alert = Alert(
            user_id=user_id,
            ticker=ticker.upper(),
            alert_type=alert_type,
            threshold=price_threshold,
            condition=condition,
            notification_channels=notification_channels or ["websocket"]
        )
        
        return await self.alert_engine.add_alert(alert)
    
    async def create_volume_alert(self, user_id: int, ticker: str, volume_threshold: float,
                                  notification_channels: List[str] = None) -> int:
        """Create volume spike alert."""
        alert = Alert(
            user_id=user_id,
            ticker=ticker.upper(),
            alert_type=AlertType.VOLUME_SPIKE,
            threshold=volume_threshold,
            condition=">",
            notification_channels=notification_channels or ["websocket"]
        )
        
        return await self.alert_engine.add_alert(alert)
    
    async def create_rsi_alert(self, user_id: int, ticker: str, rsi_threshold: float,
                               overbought: bool = True, notification_channels: List[str] = None) -> int:
        """Create RSI alert."""
        alert_type = AlertType.RSI_OVERBOUGHT if overbought else AlertType.RSI_OVERSOLD
        condition = ">" if overbought else "<"
        
        alert = Alert(
            user_id=user_id,
            ticker=ticker.upper(),
            alert_type=alert_type,
            threshold=rsi_threshold,
            condition=condition,
            notification_channels=notification_channels or ["websocket"]
        )
        
        return await self.alert_engine.add_alert(alert)
    
    async def get_user_alerts(self, user_id: int) -> List[Alert]:
        """Get all alerts for a user."""
        return [alert for alert in self.alert_engine.active_alerts.values() if alert.user_id == user_id]
    
    async def cancel_alert(self, alert_id: int, user_id: int) -> bool:
        """Cancel user's alert."""
        if alert_id in self.alert_engine.active_alerts:
            alert = self.alert_engine.active_alerts[alert_id]
            if alert.user_id == user_id:
                return await self.alert_engine.remove_alert(alert_id)
        return False
