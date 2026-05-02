"""
WebSocket handler for real-time stock price streaming.
"""
import asyncio
import json
import logging
from typing import Dict, List, Set
from datetime import datetime
import yfinance as yf
import pandas as pd
from fastapi import WebSocket, WebSocketDisconnect
import redis

logger = logging.getLogger("stockvision.websocket")

class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # connection_id -> set of tickers
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept WebSocket connection and register client."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connected",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)
    
    def disconnect(self, client_id: str):
        """Remove client connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def subscribe_ticker(self, client_id: str, ticker: str):
        """Subscribe client to ticker updates."""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].add(ticker.upper())
            await self.send_personal_message({
                "type": "subscription_added",
                "ticker": ticker.upper(),
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
            
            # Send current price immediately
            await self.send_current_price(client_id, ticker.upper())
    
    async def unsubscribe_ticker(self, client_id: str, ticker: str):
        """Unsubscribe client from ticker updates."""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(ticker.upper())
            await self.send_personal_message({
                "type": "subscription_removed",
                "ticker": ticker.upper(),
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
    
    async def send_current_price(self, client_id: str, ticker: str):
        """Send current price for a ticker."""
        try:
            # Get cached price or fetch new data
            cached_data = self.redis_client.get(f"price:{ticker}")
            if cached_data:
                price_data = json.loads(cached_data)
            else:
                # Fetch latest data
                stock = yf.Ticker(ticker)
                data = stock.history(period="1d", interval="1m")
                
                if not data.empty:
                    latest = data.iloc[-1]
                    price_data = {
                        "ticker": ticker,
                        "price": float(latest["Close"]),
                        "change": float(latest["Close"] - latest["Open"]),
                        "change_percent": float(((latest["Close"] - latest["Open"]) / latest["Open"]) * 100),
                        "volume": int(latest["Volume"]),
                        "timestamp": latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
                    }
                    
                    # Cache for 60 seconds
                    self.redis_client.setex(f"price:{ticker}", 60, json.dumps(price_data))
                else:
                    return
            
            await self.send_personal_message({
                "type": "price_update",
                "data": price_data
            }, client_id)
            
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
    
    def get_subscribed_tickers(self, client_id: str) -> Set[str]:
        """Get all tickers subscribed by a client."""
        return self.subscriptions.get(client_id, set())
    
    def get_all_subscribed_tickers(self) -> Set[str]:
        """Get all unique tickers subscribed by any client."""
        all_tickers = set()
        for tickers in self.subscriptions.values():
            all_tickers.update(tickers)
        return all_tickers

# Global connection manager
manager = ConnectionManager()

class PriceStreamer:
    """Background task to stream real-time prices."""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.running = False
        self.update_interval = 30  # seconds
    
    async def start_streaming(self):
        """Start the price streaming background task."""
        self.running = True
        logger.info("Price streaming started")
        
        while self.running:
            try:
                await self.update_prices()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in price streaming: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def update_prices(self):
        """Update prices for all subscribed tickers."""
        all_tickers = self.connection_manager.get_all_subscribed_tickers()
        
        if not all_tickers:
            return
        
        try:
            # Fetch data for all tickers at once
            tickers_str = " ".join(all_tickers)
            data = yf.download(tickers_str, period="1d", interval="1m")
            
            for ticker in all_tickers:
                try:
                    if ticker in data.columns.get_level_values(0):
                        ticker_data = data[ticker]
                        if not ticker_data.empty:
                            latest = ticker_data.iloc[-1]
                            
                            price_data = {
                                "ticker": ticker,
                                "price": float(latest["Close"]),
                                "change": float(latest["Close"] - latest["Open"]),
                                "change_percent": float(((latest["Close"] - latest["Open"]) / latest["Open"]) * 100),
                                "volume": int(latest["Volume"]),
                                "timestamp": latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
                            }
                            
                            # Cache the data
                            manager.redis_client.setex(f"price:{ticker}", 60, json.dumps(price_data))
                            
                            # Send to subscribed clients
                            await self.broadcast_to_subscribers(ticker, price_data)
                            
                except Exception as e:
                    logger.error(f"Error processing data for {ticker}: {e}")
                    
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
    
    async def broadcast_to_subscribers(self, ticker: str, price_data: dict):
        """Broadcast price update to clients subscribed to the ticker."""
        message = {
            "type": "price_update",
            "data": price_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for client_id, subscribed_tickers in manager.subscriptions.items():
            if ticker in subscribed_tickers:
                await manager.send_personal_message(message, client_id)
    
    def stop(self):
        """Stop the price streaming."""
        self.running = False
        logger.info("Price streaming stopped")

# Global price streamer
price_streamer = None

async def handle_websocket_messages(websocket: WebSocket, client_id: str):
    """Handle incoming WebSocket messages."""
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "subscribe":
                ticker = message.get("ticker")
                if ticker:
                    await manager.subscribe_ticker(client_id, ticker)
                    
            elif message_type == "unsubscribe":
                ticker = message.get("ticker")
                if ticker:
                    await manager.unsubscribe_ticker(client_id, ticker)
                    
            elif message_type == "get_subscriptions":
                subscriptions = manager.get_subscribed_tickers(client_id)
                await manager.send_personal_message({
                    "type": "subscriptions",
                    "tickers": list(subscriptions),
                    "timestamp": datetime.utcnow().isoformat()
                }, client_id)
                
            elif message_type == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }, client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Error handling WebSocket message for {client_id}: {e}")
        manager.disconnect(client_id)
