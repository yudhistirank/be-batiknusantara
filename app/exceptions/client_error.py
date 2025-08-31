class ClientError(Exception):
    def __init__(self, message="Client error occurred", status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
