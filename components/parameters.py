

class Parameters:
	""" Container class for user-specified parameters. 
	    All attributes should be supplied as ctor args. """

	def __init__(self,
				 m: int, 
				 n: int, 
				 k: int, 
				 lda: int, 
				 ldb: int, 
				 ldc: int,
				 mtx_filename: str,
				 mtx_format: str = 'any',
				 output_funcname: str = None,
				 output_filename: str = None,
				 bm: int = None, 
				 bn: int = None, 
				 bk: int = None,
				 v_size: int = None,
				 arch: str = 'knl',
				 **kwargs  # Accept and ignore args which don't belong
				 ) -> None:

		self.m = m
		self.n = n
		self.k = k

		self.lda = lda
		self.ldb = ldb
		self.ldc = ldc

		self.bm = bm
		self.bn = bn
		self.bk = bk

		self.v_size = v_size
		self.arch = arch

		self.mtx_filename = mtx_filename
		self.mtx_format = mtx_format

		self.output_funcname = output_funcname
		self.output_filename = output_filename


	def copy(self, other):
		""" Updates contents """
		Parameters.__init__(self, **other.__dict__)

	

