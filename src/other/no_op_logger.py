class NoOpLogger:
	"""Logger that does nothing - all methods are no-ops"""

	def debug(self, msg: object, *args: object, **kwargs: object) -> None:
		pass

	def info(self, msg: object, *args: object, **kwargs: object) -> None:
		pass

	def warning(self, msg: object, *args: object, **kwargs: object) -> None:
		pass

	def error(self, msg: object, *args: object, **kwargs: object) -> None:
		pass

	def critical(self, msg: object, *args: object, **kwargs: object) -> None:
		pass
