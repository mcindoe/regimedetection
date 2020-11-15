def write_maximal_eigengaps_from_W_D(
    similarities_matrix,
    min_t,
    max_t,
    folder_path,
    filename,
    logscale=True,
    n_points=1000
    ):


    similarities_row_sums = similarities_matrix.sum(axis=1)
    transition_matrix = similarities_matrix / similarities_row_sums[:, np.newaxis]

    evalues, _ = np.linalg.eig(transition_matrix)
    idx = evalues.argsort()[::-1]
    evalues = evalues[idx]

    if logscale:
        t_values = np.logspace(start=min_t, stop=max_t, num=n_points)
    else:
        t_values = range(min_t, max_t+1)

    output = []

    for t in t_values:
        val = maximal_eigengap(t, evalues)
        entry = f'{t} {val}'
        if t != max_t:
            entry += '\n'
        output.append(entry)

    with open(folder_path+filename, 'w+') as fp:
        fp.writelines(output)


def write_maximal_eigengaps(points, metric, similarity,
                            min_t, max_t, folder_path, filename,
                            logscale=True, n_points=1000):

    W, D = make_W_D(points, metric, similarity)

    write_maximal_eigengaps_from_W_D(
        W, D, min_t, max_t, folder_path,
        filename, logscale, n_points
    )


def get_eigengap_values(t_values, evalues, max_clusters=None, ignore_one_clustering=False):
    eigengap_values = {}
    max_n_clusters = len(evalues) - 1

    if max_clusters is not None:
        max_n_clusters = min(max_n_clusters, max_clusters)

    clusters = range(1, max_n_clusters)
    if ignore_one_clustering:
        clusters = range(2, max_n_clusters)

    for k in clusters:
        eigengap_values[k] = {t: delta_k_t(k-1, t, evalues) for t in t_values}

    return eigengap_values


def get_maximal_eigengap_information(eigengap_values):
    ex_key = list(eigengap_values.keys())[0]
    clusters = eigengap_values.keys()
    t_values = eigengap_values[ex_key].keys()

    max_eigengap_values = {}
    maximum_attained = {}

    for t in t_values:
        vals = {k: eigengap_values[k][t] for k in clusters}.items()
        cluster, maximum = max(vals, key=lambda x: x[1])
        if cluster not in maximum_attained:
            maximum_attained[cluster] = {
                'suitability': maximum,
                'n_steps': int(t),
            }
        else:
            if maximum > maximum_attained[cluster]['suitability']:
                maximum_attained[cluster] = {
                    'suitability': maximum,
                    'n_steps': int(t),
                }

        max_eigengap_values[t] = maximum

    return max_eigengap_values, maximum_attained


def multiscale_k_prototypes_from_W_D(points, W, D, max_clusters=None):
    transition_matrix = np.linalg.inv(D).dot(W)

    evalues, evectors = np.linalg.eig(transition_matrix)
    evectors = evectors.T
    evectors = np.array([evectors[n] for n in range(len(evectors))])

    # Sort eigenvalues from largest to smallest; update eigenvectors
    idx = evalues.argsort()[::-1]
    evalues = evalues[idx]
    evectors = evectors[idx]

    t_values = np.logspace(start=0, stop=14, num=1000)
    eigengap_values = get_eigengap_values(t_values, evalues, max_clusters)
    max_egap_values, max_attained = get_maximal_eigengap_information(eigengap_values)

    results = []

    for n_clusters in max_attained:
        if n_clusters == 1:
            continue

        n_steps = max_attained[n_clusters]['n_steps']
        suitability = max_attained[n_clusters]['suitability']

        transition_matrix_power = np.linalg.matrix_power(transition_matrix, n_steps)

        Q_init = star_shaped_init(transition_matrix_power, n_clusters)
        partition, Q = K_prototypes(transition_matrix_power, n_clusters, Q_init)
        results_entry = dict(
            n_clusters = n_clusters,
            n_steps = n_steps,
            suitability = suitability,
            partition = partition,
        )
        results.append(results_entry)

    return results


def multiscale_k_prototypes(points, metric, similarity, max_clusters=None):
    '''
    Args:
    As in make_W_D()
    '''

    W, D = make_W_D(points, metric, similarity)
    return multiscale_k_prototypes_from_W_D(points, W, D, max_clusters)

