def init_matrix(row, col):
    '''initialize from uniform distribution, divided by sqrt(col) for reasons I'm not sure of '''
    v= 1.0 /np.sqrt(col)
    result = np.random.uniform(low=(-1*v), high=v, size=(row,col) )
    return result
