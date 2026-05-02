/**
 * StockVision Progressive Web App Service Worker
 */

const CACHE_NAME = 'stockvision-v1.0.0';
const STATIC_CACHE = 'stockvision-static-v1';
const DYNAMIC_CACHE = 'stockvision-dynamic-v1';

// Assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/static/manifest.json',
  '/static/css/main.css',
  '/static/js/advanced_charts.js',
  '/static/js/app.js',
  '/static/icons/icon-192x192.png',
  '/static/icons/icon-512x512.png',
  'https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js',
  'https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.min.js'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('Service Worker: Installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('Service Worker: Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('Service Worker: Static assets cached');
        return self.skipWaiting();
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activating...');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
              console.log('Service Worker: Deleting old cache', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('Service Worker: Activated');
        return self.clients.claim();
      })
  );
});

// Fetch event - serve from cache when offline
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Handle different request types
  if (request.method === 'GET') {
    // Static assets - cache first strategy
    if (isStaticAsset(url)) {
      event.respondWith(cacheFirst(request));
    }
    // API calls - network first strategy
    else if (isAPIRequest(url)) {
      event.respondWith(networkFirst(request));
    }
    // Other requests - stale while revalidate
    else {
      event.respondWith(staleWhileRevalidate(request));
    }
  }
  // Handle POST requests for offline functionality
  else if (request.method === 'POST') {
    event.respondWith(handlePostRequest(request));
  }
});

// Cache First Strategy
async function cacheFirst(request) {
  try {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.error('Cache First failed:', error);
    return new Response('Offline - No cached data available', {
      status: 503,
      statusText: 'Service Unavailable'
    });
  }
}

// Network First Strategy
async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log('Network failed, trying cache:', error);
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline fallback for API requests
    if (request.url.includes('/predict')) {
      return new Response(JSON.stringify({
        error: 'Offline - Using cached predictions',
        data: await getCachedPredictions()
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    return new Response('Offline - No network connection', {
      status: 503,
      statusText: 'Service Unavailable'
    });
  }
}

// Stale While Revalidate Strategy
async function staleWhileRevalidate(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  const cachedResponse = await cache.match(request);
  
  const fetchPromise = fetch(request).then((networkResponse) => {
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  });
  
  return cachedResponse || fetchPromise;
}

// Handle POST requests for offline functionality
async function handlePostRequest(request) {
  try {
    const networkResponse = await fetch(request);
    return networkResponse;
  } catch (error) {
    console.log('POST request failed, storing for later:', error);
    
    // Store the request for when we're back online
    const requestData = await request.clone().text();
    await storeOfflineRequest(request.url, requestData, request.headers);
    
    return new Response(JSON.stringify({
      success: false,
      message: 'Request stored for when you\'re back online'
    }), {
      status: 202,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

// Helper functions
function isStaticAsset(url) {
  return url.pathname.startsWith('/static/') ||
         url.pathname === '/' ||
         url.pathname.includes('.js') ||
         url.pathname.includes('.css') ||
         url.pathname.includes('.png') ||
         url.pathname.includes('.jpg') ||
         url.pathname.includes('.ico');
}

function isAPIRequest(url) {
  return url.pathname.includes('/api/') ||
         url.pathname.includes('/predict') ||
         url.pathname.includes('/ws');
}

// Store offline requests in IndexedDB
async function storeOfflineRequest(url, data, headers) {
  if (!self.indexedDB) return;
  
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('StockVisionOfflineDB', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction(['offlineRequests'], 'readwrite');
      const store = transaction.objectStore('offlineRequests');
      
      store.add({
        url,
        data,
        headers: Object.fromEntries(headers.entries()),
        timestamp: Date.now()
      });
      
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    };
    
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('offlineRequests')) {
        db.createObjectStore('offlineRequests', { keyPath: 'id', autoIncrement: true });
      }
    };
  });
}

// Get cached predictions for offline fallback
async function getCachedPredictions() {
  try {
    const cache = await caches.open(DYNAMIC_CACHE);
    const cachedResponse = await cache.match('/api/predictions/history');
    if (cachedResponse) {
      return await cachedResponse.json();
    }
  } catch (error) {
    console.error('Failed to get cached predictions:', error);
  }
  return [];
}

// Background sync for offline requests
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    event.waitUntil(syncOfflineRequests());
  }
});

async function syncOfflineRequests() {
  if (!self.indexedDB) return;
  
  try {
    const db = await openIndexedDB();
    const requests = await getAllOfflineRequests(db);
    
    for (const request of requests) {
      try {
        await fetch(request.url, {
          method: 'POST',
          headers: request.headers,
          body: request.data
        });
        
        // Remove successful request from storage
        await deleteOfflineRequest(db, request.id);
      } catch (error) {
        console.error('Failed to sync request:', error);
      }
    }
  } catch (error) {
    console.error('Background sync failed:', error);
  }
}

// Push notifications
self.addEventListener('push', (event) => {
  const options = {
    body: event.data ? event.data.text() : 'New stock alert available',
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'Explore',
        icon: '/static/icons/checkmark.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/static/icons/xmark.png'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification('StockVision Alert', options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// IndexedDB helpers
function openIndexedDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('StockVisionOfflineDB', 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('offlineRequests')) {
        db.createObjectStore('offlineRequests', { keyPath: 'id', autoIncrement: true });
      }
    };
  });
}

async function getAllOfflineRequests(db) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['offlineRequests'], 'readonly');
    const store = transaction.objectStore('offlineRequests');
    const request = store.getAll();
    
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function deleteOfflineRequest(db, id) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['offlineRequests'], 'readwrite');
    const store = transaction.objectStore('offlineRequests');
    const request = store.delete(id);
    
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// Performance monitoring
self.addEventListener('fetch', (event) => {
  if (event.request.url.includes('/api/')) {
    const start = performance.now();
    
    event.respondWith(
      fetch(event.request).then(response => {
        const duration = performance.now() - start;
        console.log(`API Request took ${duration.toFixed(2)}ms: ${event.request.url}`);
        
        // Store performance metrics
        storePerformanceMetric(event.request.url, duration);
        
        return response;
      })
    );
  }
});

async function storePerformanceMetric(url, duration) {
  try {
    const cache = await caches.open(DYNAMIC_CACHE);
    const metrics = await cache.match('/api/metrics');
    let data = metrics ? await metrics.json() : { requests: [] };
    
    data.requests.push({
      url,
      duration,
      timestamp: Date.now()
    });
    
    // Keep only last 100 metrics
    if (data.requests.length > 100) {
      data.requests = data.requests.slice(-100);
    }
    
    const response = new Response(JSON.stringify(data), {
      headers: { 'Content-Type': 'application/json' }
    });
    
    await cache.put('/api/metrics', response);
  } catch (error) {
    console.error('Failed to store performance metric:', error);
  }
}
