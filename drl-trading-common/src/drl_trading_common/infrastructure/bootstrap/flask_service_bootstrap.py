import os
import logging

from drl_trading_common.infrastructure.bootstrap.service_bootstrap import (
    ServiceBootstrap,
)

logger = logging.getLogger(__name__)


class FlaskServiceBootstrap(ServiceBootstrap):
    """
    Specialized bootstrap for services that require Flask web interface.

    Automatically enables web interface and provides Flask-specific main loop.
    """

    def enable_web_interface(self) -> bool:
        """Flask services always enable web interface."""
        return True

    def _run_main_loop(self) -> None:
        """Run Flask development server as main loop."""
        if self._flask_app:
            # In production, use a proper WSGI server like gunicorn
            # For development, run directly
            port = int(os.environ.get("PORT", 8080))
            debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

            logger.info(f"Starting Flask application on port {port}")
            self._flask_app.run(
                host="0.0.0.0", port=port, debug=debug_mode, threaded=True
            )
        else:
            logger.error("No Flask application to run")
            # Keep the service alive if no app
            import time

            while self.is_running:
                time.sleep(1)
