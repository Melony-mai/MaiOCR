from PySide6.QtNetwork import QLocalServer, QLocalSocket

from maiocr.utils.logger import get_logger

logger = get_logger()

SERVER_KEY = "maiocr-single-instance"


class SingleInstance:
    """
    Cross-platform single-instance guard based on QLocalServer.

    Usage:
        guard = SingleInstance()
        if not guard.acquire():
            return  # another instance is running
    """

    def __init__(self, key: str = SERVER_KEY):
        self._key = key
        self._server: QLocalServer | None = None

    def acquire(self) -> bool:
        probe = QLocalSocket()
        probe.connectToServer(self._key)

        if probe.waitForConnected(300):
            logger.info("Another MaiOCR instance detected, exiting")
            probe.disconnectFromServer()
            return False

        QLocalServer.removeServer(self._key)

        self._server = QLocalServer()
        if not self._server.listen(self._key):
            logger.error(
                "Single-instance server failed: {}",
                self._server.errorString(),
            )
            return False

        # Accept (and drop) connections from duplicate launches.
        def _on_new_connection():
            if self._server is not None:
                client = self._server.nextPendingConnection()
                if client is not None:
                    client.disconnectFromServer()
                    client.deleteLater()

        self._server.newConnection.connect(_on_new_connection)
        logger.info("Single-instance lock acquired")
        return True

    def release(self):
        if self._server is not None:
            self._server.close()
            self._server = None
