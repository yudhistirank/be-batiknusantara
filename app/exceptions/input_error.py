class InputError(Exception):
    def __init__(self, message="Invalid input", status_code=422):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
