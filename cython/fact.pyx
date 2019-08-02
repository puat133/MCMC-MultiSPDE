def typed_fact(long n):
    """Computes n!"""
    if n <= 1:
        return 1
    return n * typed_fact(n - 1)

cpdef long c_fact(long n):
    """Computes n!"""
    if n <= 1:
        return 1
    return n * c_fact(n - 1)

