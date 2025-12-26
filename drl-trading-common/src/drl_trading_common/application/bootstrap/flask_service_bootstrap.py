import logging
import sys

from drl_trading_common.application.bootstrap.service_bootstrap import (
    ServiceBootstrap,
)

logger = logging.getLogger(__name__)


class FlaskServiceBootstrap(ServiceBootstrap):
    """
    Specialized bootstrap for services that require Flask web interface.

    Automatically enables web interface and provides Flask-specific main loop
    with proper shutdown handling.
    """

    def __init__(self, service_name: str, config_class: type) -> None:
        """Initialize Flask service bootstrap with shutdown handling."""
        super().__init__(service_name, config_class)
        self._flask_server_thread = None

    def enable_web_interface(self) -> bool:
        """Flask services always enable web interface."""
        return True

    def _signal_handler(self, signum: int, frame: object) -> None:
        """Override signal handler to properly shutdown Flask."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
        # Force exit if Flask doesn't shutdown gracefully
        if self._flask_app:
            logger.info("Forcing Flask application shutdown...")
            sys.exit(0)

    def _run_main_loop(self) -> None:
        """Run Flask development server as main loop with proper shutdown."""
        if self._flask_app:
            # Get WebAPI configuration from the service config
            infrastructure = getattr(self.config, 'infrastructure', None)
            webapi_config = getattr(infrastructure, 'webapi', None)

            if webapi_config and hasattr(webapi_config, 'port'):
                port = webapi_config.port
            else:
                port = 8080
                logger.warning("No WebAPI port configuration found, using default port 8080")

            if webapi_config and hasattr(webapi_config, 'debug'):
                debug = webapi_config.debug
            else:
                debug = False
                logger.warning("No WebAPI debug configuration found, using default debug False")

            logger.info(f"Starting Flask application on port {port}")
            try:
                self._flask_app.run(
                    host="0.0.0.0",
                    port=port,
                    debug=debug,
                    threaded=True,
                    use_reloader=False
                )
            except KeyboardInterrupt:
                logger.info("Flask application interrupted by user")
            except Exception as e:
                logger.error(f"Flask application error: {e}")
        else:
            logger.error("No Flask application to run")
            # Keep the service alive if no app
            import time

            while self.is_running:
                time.sleep(1)
