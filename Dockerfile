FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
        gnupg \
            ca-certificates \
                fonts-liberation \
                    libasound2 \
                        libatk-bridge2.0-0 \
                            libatk1.0-0 \
                                libatspi2.0-0 \
                                    libcups2 \
                                        libdbus-1-3 \
                                            libdrm2 \
                                                libgbm1 \
                                                    libgtk-3-0 \
                                                        libnspr4 \
                                                            libnss3 \
                                                                libwayland-client0 \
                                                                    libxcomposite1 \
                                                                        libxdamage1 \
                                                                            libxfixes3 \
                                                                                libxkbcommon0 \
                                                                                    libxrandr2 \
                                                                                        xdg-utils \
                                                                                            xvfb \
                                                                                                x11vnc \
                                                                                                    novnc \
                                                                                                        websockify \
                                                                                                            nginx \
                                                                                                                supervisor \
                                                                                                                    procps \
                                                                                                                        && rm -rf /var/lib/apt/lists/*

                                                                                                                        # Copy requirements file
                                                                                                                        COPY requirements.txt .

                                                                                                                        # Install Python dependencies
                                                                                                                        RUN pip install --no-cache-dir -r requirements.txt

                                                                                                                        # Install Playwright browsers
                                                                                                                        RUN python -m playwright install chromium

                                                                                                                        # Copy application code
                                                                                                                        COPY main.py .

                                                                                                                        # Copy nginx configuration
                                                                                                                        COPY nginx.conf /etc/nginx/nginx.conf

                                                                                                                        # Copy supervisor configuration
                                                                                                                        COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

                                                                                                                        # Create necessary directories
                                                                                                                        RUN mkdir -p /var/log/supervisor /var/run/supervisor

                                                                                                                        # Expose only one port (8080 for Azure Container Apps)
                                                                                                                        EXPOSE 8080

                                                                                                                        # Set environment variables
                                                                                                                        ENV PYTHONUNBUFFERED=1
                                                                                                                        ENV DISPLAY=:99

                                                                                                                        # Use supervisor to manage all processes
                                                                                                                        CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]