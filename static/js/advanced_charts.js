/**
 * Advanced D3.js visualization components for StockVision
 */
class AdvancedCharts {
    constructor() {
        this.margin = { top: 20, right: 30, bottom: 40, left: 50 };
        this.colors = {
            bullish: '#00ff88',
            bearish: '#ff4444',
            neutral: '#ffaa00',
            volume: '#6b46c1',
            sma: '#3b82f6',
            ema: '#10b981',
            rsi: '#f59e0b'
        };
    }

    // Candlestick Chart
    createCandlestickChart(containerId, data, options = {}) {
        const container = d3.select(`#${containerId}`);
        container.selectAll("*").remove();

        const width = options.width || container.node().offsetWidth;
        const height = options.height || 400;
        const innerWidth = width - this.margin.left - this.margin.right;
        const innerHeight = height - this.margin.top - this.margin.bottom;

        const svg = container.append("svg")
            .attr("width", width)
            .attr("height", height);

        const g = svg.append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);

        // Scales
        const xScale = d3.scaleTime()
            .domain(d3.extent(data, d => d.date))
            .range([0, innerWidth]);

        const yScale = d3.scaleLinear()
            .domain([
                d3.min(data, d => d.low) * 0.98,
                d3.max(data, d => d.high) * 1.02
            ])
            .range([innerHeight, 0]);

        // Volume scale
        const volumeScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.volume)])
            .range([innerHeight, innerHeight - 100]);

        // Create candlesticks
        const candlesticks = g.selectAll(".candlestick")
            .data(data)
            .enter().append("g")
            .attr("class", "candlestick")
            .attr("transform", d => `translate(${xScale(d.date)},0)`);

        // High-low lines
        candlesticks.append("line")
            .attr("class", "high-low")
            .attr("x1", 0)
            .attr("x2", 0)
            .attr("y1", d => yScale(d.high))
            .attr("y2", d => yScale(d.low))
            .attr("stroke", d => d.close > d.open ? this.colors.bullish : this.colors.bearish)
            .attr("stroke-width", 1);

        // Open-close rectangles
        candlesticks.append("rect")
            .attr("class", "open-close")
            .attr("x", -5)
            .attr("width", 10)
            .attr("y", d => yScale(Math.max(d.open, d.close)))
            .attr("height", d => Math.abs(yScale(d.open) - yScale(d.close)))
            .attr("fill", d => d.close > d.open ? this.colors.bullish : this.colors.bearish)
            .attr("stroke", d => d.close > d.open ? this.colors.bullish : this.colors.bearish)
            .attr("stroke-width", 1);

        // Volume bars
        const volumeBars = g.selectAll(".volume")
            .data(data)
            .enter().append("rect")
            .attr("class", "volume")
            .attr("x", d => xScale(d.date) - 5)
            .attr("width", 10)
            .attr("y", d => volumeScale(d.volume))
            .attr("height", d => innerHeight - volumeScale(d.volume))
            .attr("fill", this.colors.volume)
            .attr("opacity", 0.3);

        // Technical indicators
        if (options.showSMA && data[0].sma) {
            const smaLine = d3.line()
                .x(d => xScale(d.date))
                .y(d => yScale(d.sma))
                .curve(d3.curveMonotoneX);

            g.append("path")
                .datum(data)
                .attr("class", "sma-line")
                .attr("d", smaLine)
                .attr("fill", "none")
                .attr("stroke", this.colors.sma)
                .attr("stroke-width", 2);
        }

        if (options.showEMA && data[0].ema) {
            const emaLine = d3.line()
                .x(d => xScale(d.date))
                .y(d => yScale(d.ema))
                .curve(d3.curveMonotoneX);

            g.append("path")
                .datum(data)
                .attr("class", "ema-line")
                .attr("d", emaLine)
                .attr("fill", "none")
                .attr("stroke", this.colors.ema)
                .attr("stroke-width", 2);
        }

        // Axes
        const xAxis = d3.axisBottom(xScale)
            .tickFormat(d3.timeFormat("%m/%d"));

        const yAxis = d3.axisLeft(yScale)
            .tickFormat(d3.format("$.2f"));

        g.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0,${innerHeight})`)
            .call(xAxis);

        g.append("g")
            .attr("class", "y-axis")
            .call(yAxis);

        // Tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0)
            .style("position", "absolute")
            .style("background", "rgba(0,0,0,0.8)")
            .style("color", "white")
            .style("padding", "8px")
            .style("border-radius", "4px")
            .style("font-size", "12px");

        candlesticks.on("mouseover", function(event, d) {
            tooltip.transition()
                .duration(200)
                .style("opacity", .9);
            tooltip.html(`
                <strong>${d.date.toLocaleDateString()}</strong><br/>
                Open: $${d.open.toFixed(2)}<br/>
                High: $${d.high.toFixed(2)}<br/>
                Low: $${d.low.toFixed(2)}<br/>
                Close: $${d.close.toFixed(2)}<br/>
                Volume: ${d.volume.toLocaleString()}
            `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function() {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        });

        // Zoom and pan
        const zoom = d3.zoom()
            .scaleExtent([0.5, 10])
            .on("zoom", (event) => {
                const newXScale = event.transform.rescaleX(xScale);
                const newYScale = event.transform.rescaleY(yScale);

                g.select(".x-axis").call(xAxis.scale(newXScale));
                g.select(".y-axis").call(yAxis.scale(newYScale));

                candlesticks.attr("transform", d => `translate(${newXScale(d.date)},0)`);
                candlesticks.select(".high-low")
                    .attr("y1", d => newYScale(d.high))
                    .attr("y2", d => newYScale(d.low));
                candlesticks.select(".open-close")
                    .attr("y", d => newYScale(Math.max(d.open, d.close)))
                    .attr("height", d => Math.abs(newYScale(d.open) - newYScale(d.close)));
            });

        svg.call(zoom);
    }

    // Heatmap for correlation matrix
    createCorrelationHeatmap(containerId, data, tickers) {
        const container = d3.select(`#${containerId}`);
        container.selectAll("*").remove();

        const width = container.node().offsetWidth;
        const height = width;
        const cellSize = width / tickers.length;

        const svg = container.append("svg")
            .attr("width", width)
            .attr("height", height);

        const colorScale = d3.scaleSequential(d3.interpolateRdBu)
            .domain([-1, 1]);

        // Create cells
        const cells = svg.selectAll(".cell")
            .data(data)
            .enter().append("g")
            .attr("class", "cell")
            .attr("transform", (d, i) => {
                const row = Math.floor(i / tickers.length);
                const col = i % tickers.length;
                return `translate(${col * cellSize},${row * cellSize})`;
            });

        cells.append("rect")
            .attr("width", cellSize)
            .attr("height", cellSize)
            .attr("fill", d => colorScale(d.correlation))
            .attr("stroke", "white")
            .attr("stroke-width", 1);

        // Add correlation values
        cells.append("text")
            .attr("x", cellSize / 2)
            .attr("y", cellSize / 2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("font-size", "10px")
            .attr("fill", d => Math.abs(d.correlation) > 0.5 ? "white" : "black")
            .text(d => d.correlation.toFixed(2));

        // Add ticker labels
        const labelScale = d3.scaleBand()
            .domain(tickers)
            .range([0, width]);

        // X-axis labels
        svg.selectAll(".x-label")
            .data(tickers)
            .enter().append("text")
            .attr("class", "x-label")
            .attr("x", (d, i) => (i + 0.5) * cellSize)
            .attr("y", height - 5)
            .attr("text-anchor", "middle")
            .attr("font-size", "12px")
            .text(d => d);

        // Y-axis labels
        svg.selectAll(".y-label")
            .data(tickers)
            .enter().append("text")
            .attr("class", "y-label")
            .attr("x", 5)
            .attr("y", (d, i) => (i + 0.5) * cellSize)
            .attr("text-anchor", "start")
            .attr("font-size", "12px")
            .text(d => d);
    }

    // 3D Portfolio Visualization
    create3DPortfolio(containerId, portfolioData) {
        const container = d3.select(`#${containerId}`);
        container.selectAll("*").remove();

        const width = container.node().offsetWidth;
        const height = 400;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });

        renderer.setSize(width, height);
        container.node().appendChild(renderer.domElement);

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        // Create portfolio spheres
        const geometry = new THREE.SphereGeometry(1, 32, 32);
        
        portfolioData.forEach((stock, i) => {
            const material = new THREE.MeshPhongMaterial({
                color: stock.performance > 0 ? 0x00ff88 : 0xff4444
            });
            
            const sphere = new THREE.Mesh(geometry, material);
            
            // Position based on performance and allocation
            const angle = (i / portfolioData.length) * Math.PI * 2;
            const radius = 5;
            sphere.position.x = Math.cos(angle) * radius;
            sphere.position.y = stock.performance * 10;
            sphere.position.z = Math.sin(angle) * radius;
            
            // Scale based on allocation
            const scale = 0.5 + stock.allocation * 2;
            sphere.scale.set(scale, scale, scale);
            
            scene.add(sphere);
        });

        camera.position.z = 15;

        // Animation
        function animate() {
            requestAnimationFrame(animate);
            
            scene.rotation.y += 0.005;
            
            renderer.render(scene, camera);
        }
        
        animate();

        // Handle resize
        window.addEventListener('resize', () => {
            const newWidth = container.node().offsetWidth;
            camera.aspect = newWidth / height;
            camera.updateProjectionMatrix();
            renderer.setSize(newWidth, height);
        });
    }

    // Real-time streaming chart
    createStreamingChart(containerId, initialData = []) {
        const container = d3.select(`#${containerId}`);
        container.selectAll("*").remove();

        const width = container.node().offsetWidth;
        const height = 300;
        const maxDataPoints = 100;

        const svg = container.append("svg")
            .attr("width", width)
            .attr("height", height);

        const margin = { top: 20, right: 30, bottom: 30, left: 50 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const xScale = d3.scaleTime()
            .range([0, innerWidth]);

        const yScale = d3.scaleLinear()
            .range([innerHeight, 0]);

        const line = d3.line()
            .x(d => xScale(d.timestamp))
            .y(d => yScale(d.price))
            .curve(d3.curveMonotoneX);

        let data = initialData;

        function updateChart(newData) {
            data = data.concat(newData).slice(-maxDataPoints);

            xScale.domain(d3.extent(data, d => d.timestamp));
            yScale.domain([
                d3.min(data, d => d.price) * 0.999,
                d3.max(data, d => d.price) * 1.001
            ]);

            const path = g.selectAll(".price-line")
                .data([data]);

            path.enter()
                .append("path")
                .attr("class", "price-line")
                .merge(path)
                .attr("d", line)
                .attr("fill", "none")
                .attr("stroke", this.colors.bullish)
                .attr("stroke-width", 2);

            // Update axes
            g.select(".x-axis").remove();
            g.select(".y-axis").remove();

            g.append("g")
                .attr("class", "x-axis")
                .attr("transform", `translate(0,${innerHeight})`)
                .call(d3.axisBottom(xScale).ticks(5));

            g.append("g")
                .attr("class", "y-axis")
                .call(d3.axisLeft(yScale).ticks(5));
        }

        return { updateChart };
    }

    // Technical indicator chart
    createTechnicalIndicatorChart(containerId, data, indicatorType) {
        const container = d3.select(`#${containerId}`);
        container.selectAll("*").remove();

        const width = container.node().offsetWidth;
        const height = 200;

        const svg = container.append("svg")
            .attr("width", width)
            .attr("height", height);

        const margin = { top: 10, right: 30, bottom: 30, left: 50 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;

        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const xScale = d3.scaleTime()
            .domain(d3.extent(data, d => d.date))
            .range([0, innerWidth]);

        let yScale, line, area;

        switch (indicatorType) {
            case 'rsi':
                yScale = d3.scaleLinear().domain([0, 100]).range([innerHeight, 0]);
                line = d3.line()
                    .x(d => xScale(d.date))
                    .y(d => yScale(d.rsi))
                    .curve(d3.curveMonotoneX);
                
                // Add overbought/oversold lines
                g.append("line")
                    .attr("x1", 0).attr("x2", innerWidth)
                    .attr("y1", yScale(70)).attr("y2", yScale(70))
                    .attr("stroke", this.colors.bearish).attr("stroke-dash", "5,5");
                
                g.append("line")
                    .attr("x1", 0).attr("x2", innerWidth)
                    .attr("y1", yScale(30)).attr("y2", yScale(30))
                    .attr("stroke", this.colors.bullish).attr("stroke-dash", "5,5");
                break;

            case 'volume':
                yScale = d3.scaleLinear()
                    .domain([0, d3.max(data, d => d.volume)])
                    .range([innerHeight, 0]);
                
                area = d3.area()
                    .x(d => xScale(d.date))
                    .y0(innerHeight)
                    .y1(d => yScale(d.volume));
                break;

            case 'macd':
                const macdValues = data.flatMap(d => [d.macd, d.signal, d.histogram]);
                yScale = d3.scaleLinear()
                    .domain(d3.extent(macdValues))
                    .range([innerHeight, 0]);
                
                // MACD line
                const macdLine = d3.line()
                    .x(d => xScale(d.date))
                    .y(d => yScale(d.macd));
                
                // Signal line
                const signalLine = d3.line()
                    .x(d => xScale(d.date))
                    .y(d => yScale(d.signal));
                
                g.append("path").datum(data).attr("d", macdLine)
                    .attr("fill", "none").attr("stroke", "#3b82f6").attr("stroke-width", 2);
                
                g.append("path").datum(data).attr("d", signalLine)
                    .attr("fill", "none").attr("stroke", "#ef4444").attr("stroke-width", 2);
                break;
        }

        // Draw the main indicator
        if (indicatorType === 'volume') {
            g.append("path").datum(data).attr("d", area)
                .attr("fill", this.colors.volume).attr("opacity", 0.6);
        } else if (indicatorType === 'rsi') {
            g.append("path").datum(data).attr("d", line)
                .attr("fill", "none").attr("stroke", this.colors.rsi).attr("stroke-width", 2);
        }

        // Axes
        g.append("g")
            .attr("transform", `translate(0,${innerHeight})`)
            .call(d3.axisBottom(xScale).ticks(5));

        g.append("g")
            .call(d3.axisLeft(yScale).ticks(5));
    }
}

// Initialize charts when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.advancedCharts = new AdvancedCharts();
});
