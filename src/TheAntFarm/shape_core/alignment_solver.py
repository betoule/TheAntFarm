#import matplotlib.pyplot as plt
import cv2
import numpy as np

def euclidian(x1, y1, x2, y2):
    return np.sqrt((np.array(x1) - np.array(x2)) ** 2 + (np.array(y1) - np.array(y2)) ** 2)


class NearestNeighAssoc():
    """Solve the fixed-radius nearest neighbor search on a 2D cartesian
    lattice 

    The code uses a dictionnary to save memory so that the lattice can
    be adjusted to any search radius.
    """
    def __init__(self, first=[], extension=[], radius=1):
        self.belongs = {}
        self.clusters = []
        self.radius = radius
        
        if extension:
            xmin, xmax, ymin, ymax = extension
            self.x_bins = np.arange(xmin - 0.01 * radius, xmax + 0.01 * radius, radius)
            self.y_bins = np.arange(ymin - 0.01 * radius, ymax + 0.01 * radius, radius)
        elif first:
            firstx, firsty = first
            xmin, xmax, ymin, ymax = firstx.min(), firstx.max(), firsty.min(), firsty.max()
            self.x_bins = np.arange(xmin - 0.01 * radius, xmax + 0.01 * radius, radius)
            self.y_bins = np.arange(ymin - 0.01 * radius, ymax + 0.01 * radius, radius)
            self.clusters = list(zip(firstx, firsty))
            i = np.digitize(firstx, self.x_bins)
            j = np.digitize(firsty, self.y_bins)
            for k in range(len(i)):
                ik, jk = i[k], j[k]
                self.belongs[(ik, jk)] = self.belongs.get((ik, jk), []) + [k]
        
            
    def append(self, x, y, metric=euclidian):
        if not hasattr(self, 'x_bins'):
            xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
            self.x_bins = np.arange(xmin - 0.01 * self.radius, xmax + 0.01 * self.radius, self.radius)
            self.y_bins = np.arange(ymin - 0.01 * self.radius, ymax + 0.01 * self.radius, self.radius)

        i = np.digitize(x, self.x_bins)
        j = np.digitize(y, self.y_bins)
        index = np.zeros(len(i))
        
        for k in range(len(i)):
            ik, jk = i[k], j[k]
            # gather the list of clusters in the neighborhood
            candidates = sum([self.belongs.get((i_n, j_n), []) for i_n in (ik - 1, ik, ik + 1) for j_n in (jk - 1, jk, jk + 1)],[])
            if candidates:
                distance = metric(x[k], y[k], [self.clusters[l][0] for l in candidates], [self.clusters[l][1] for l in candidates])
                l = distance.argmin()
            if candidates and distance[l] < self.radius:
                m = candidates[l]
                index[k] = m
                self.clusters[m][2] += 1
            else:
                clu_i = len(self.clusters)
                index[k] = clu_i
                self.clusters.append([x[k], y[k], 1])
                self.belongs[(ik, jk)] = self.belongs.get((ik, jk), []) + [clu_i]
        return index

    def match(self, x, y, metric=euclidian):
        ''' Return the index of the nearest neighbor in the reference catalog

        -1 if not found
        '''
        i = np.digitize(x, self.x_bins)
        j = np.digitize(y, self.y_bins)
        index = np.zeros(len(i), dtype='int')
        for k in range(len(i)):
            ik, jk = i[k], j[k]
            # gather the list of clusters in the neighborhood
            candidates = sum([self.belongs.get((i_n, j_n), []) for i_n in (ik - 1, ik, ik + 1) for j_n in (jk - 1, jk, jk + 1)],[])
            if candidates:
                distance = metric(x[k], y[k], [self.clusters[l][0] for l in candidates], [self.clusters[l][1] for l in candidates])
                l = distance.argmin()
            if candidates and distance[l] < self.radius:
                m = candidates[l]
                index[k] = m
            else:
                index[k] = -1
        return index
    
def match(xref, yref, x, y, radius=1):
    assoc =  NearestNeighAssoc(first=[xref, yref], radius=radius)
    index = assoc.match(x, y, metric=euclidian)
    return index

def get_quad_combinations(coordinates, resolution=10, combmax=10, combmin=8, visitmax=150):
    '''Heuristic selection of quadrangles within the huge combinatorial space

    Cut the catalog in a mesh with given resolution. For each cell
    build quadrangles using candidates from the cell and the 3
    neighboring cells.
    '''
    mesh_x = np.arange(coordinates['x'].min(), coordinates['x'].max(), resolution)
    mesh_y = np.arange(coordinates['y'].min(), coordinates['y'].max(), resolution)
    i, j = np.digitize(coordinates['x'], mesh_x),  np.digitize(coordinates['y'], mesh_x)
    
    cell_map = {}
    for _i, _j, n in zip(i, j, np.arange(len(coordinates))):
        cell = cell_map.get((_i, _j), [])
        cell.append(n)
        cell_map[(_i, _j)] = cell

    quadrangles = []
    for i in range(len(mesh_x) - 1):
        for j in range(len(mesh_y) - 1):
            c1 = cell_map.get((i, j), [])
            c2 = cell_map.get((i + 1, j), [])
            c3 = cell_map.get((i, j + 1), [])
            c4 = cell_map.get((i + 1, j + 1), [])
            quadrangles.extend([(i1, i2, i3, i4) for i1 in c1 for i2 in c2 for i3 in c3 for i4 in c4])

    return np.array(quadrangles)
            
    
def get_astrometric_hash(coordinates, quadrangles, radec=False,
                         mirror=False, precision=50, iref=0, show=False,
                         debug=False, prefix=""):
    '''Compute the hash proposed in Lang et al. 2010 (see Atrometry.net)
    
    For a quadrangle A, B, C, D we compute the coordinates of C and D
    in a frame where A is the origin and B is (1, 1). This hash is
    invariant by rescaling, rotation and translation of the
    coordinates. It also became unique if one choose A and B as the
    two most distant pair and reorder the hash in a uniq way.

    Finding a transformation between two catalogs is reduce to looking
    for matches in a hash table.

    '''
    if radec:
        x,y = gp(coordinates['ra'][quadrangles]*deg2rad, coordinates['dec'][quadrangles]*deg2rad,
                                  coordinates['ra'][quadrangles[:,iref]]*deg2rad,
                                  coordinates['dec'][quadrangles[:,iref]]*deg2rad)
        x,y = x/deg2rad, y/deg2rad
    else:
        x,y = coordinates[prefix+"x"][quadrangles], coordinates[prefix+"y"][quadrangles]
    points = np.array([x, y])

    #Find the two most distant corner in the quadrangle
    index = np.array([(i, j) for i in range(0, 4) for j in range(i + 1, 4)]) # possible pairs
    distances = np.array([((points[:, :, i] - points[:, :, j]) ** 2).sum(axis=0) for i, j in index])
    i_a, i_b = index[distances.argmax(axis=0), :].T # choose the longest for AB

    # look-up table for complementary pairs
    others = np.array([[(0,0), (2, 3), (1, 3), (1, 2)], 
                       [(2,3), (1, 1), (0, 3), (0, 2)], 
                       [(1,3), (0, 3), (2, 2), (0, 1)],
                       [(1,2), (0, 2), (0, 1), (3, 3)]])
    i_c, i_d = others[i_a, i_b, :].T
    
    sl = np.arange(len(i_a))
    A = points[:, sl, i_a]
    B = points[:, sl, i_b] - A # A at the origin
    C = points[:, sl, i_c] - A
    D = points[:, sl, i_d] - A  

    # Frame rotation
    sq2 = 0.5
    Rotp = np.array([[sq2, -sq2], [sq2, sq2]])
    x = np.array(np.dot(Rotp.T , B))
    y = np.array(np.dot(Rotp , B))

    # Handles the case of inverted coordinates
    if mirror:
        xc = x.copy()
        x = y
        y = xc

    # Normalize coordinates
    nx = np.sqrt((x**2).sum(axis=0)[None,:])
    x /= nx
    ny = np.sqrt((y**2).sum(axis=0)[None,:])
    y /= ny
    nb = np.sqrt((B**2).sum(axis=0))/np.sqrt(2)
    xc, yc = (C*x).sum(axis=0)/nb , (C*y).sum(axis=0)/nb
    xd, yd = (D*x).sum(axis=0)/nb , (D*y).sum(axis=0)/nb

    if show:
        import matplotlib.pyplot as plt
        plt.figure()
        for b, c, d, _x, _y in zip(B, C, D, x*nx, y*ny):
            l = plt.plot(0, 0, 'o')
            plt.text(0, 0, 'A', c=l[0].get_color())
            for P, n in zip([b, c, d, _x, _y], 'BCDxy'):
                plt.plot( P[0], P[1], 'o', c=l[0].get_color())
                print(P[0], P[1], n)
                plt.text(P[0], P[1], n, c=l[0].get_color())
        plt.axis('equal')

    # Permutation to get unique solution
    hashk = np.vstack((xc,yc,xd,yd))
    switch_a = xc+xd>1
    hashk[:,switch_a] = 1-hashk[:,switch_a]
    switch_c = hashk[0,:]>hashk[2,:]
    hashk[:,switch_c] = hashk[:,switch_c][[2,3,0,1],:]
    hashk = np.array((precision*hashk).round(0), dtype=int)

    # Apply the relabeling to quadrangles
    i_a, i_b, i_c, i_d = np.where(switch_a, i_a, i_b), np.where(switch_a, i_b, i_a), np.where(switch_c, i_c, i_d), np.where(switch_c, i_d, i_c)
    q2 = quadrangles[sl,np.array([i_a, i_b, i_c, i_d])].T
    return hashk, q2

def plot_quadrangles(refcat, quadrangles, numerate=False, **keys):
    for i in range(quadrangles.shape[0]):
        s = refcat[quadrangles[i,:]]
        plt.plot(s['x'], s['y'], **keys)
        if numerate:
            plt.text(s['x'].mean(), s['y'].mean(), '%d' % i)

def quad_search(coordinates, hash_map, precision=50):
    nobjects = np.min([len(coordinates), 30])
    
    q2 = np.array([(i, j, k, l) for i in range(nobjects)
                   for j in range(i+1, nobjects)
                   for k in range(j+1, nobjects)
                   for l in range(k+1, nobjects)])
    hashes, kuadrangles = get_astrometric_hash(coordinates, q2, precision=precision)
    matches = []
    for h, k in zip(hashes.T, kuadrangles):
        if tuple(h) in hash_map:
            matches.append([h, k, hash_map[tuple(h)]])
    return matches

def quad_fit(refcat, cat, sky, pix, astropy=False, prefix="g", full_cat=None):
        """From match to fit. 
        
        This is the default routine used to match a linear
        transformation from a pair of quadrangle. It return a WCS_CCD
        object which contains all the material computed from the fit
        to convert ccd coordinate into sky coordinates.
        If astropy is True, an astropy.wcs object is returned instead. 
        """
        wc_cand = candidate_to_transfo(sky, pix, astropy=astropy, prefix=prefix)
        if astropy:
            cra, cdec = wc_cand.all_pix2world(np.array([cat[prefix+'x'], cat[prefix+'y']]).T, ALL_P2W).T
        else:
            cra, cdec = wc_cand.pix2world(cat[prefix+'x'], cat[prefix+'y'])
        index = match(refcat, np.rec.fromarrays([cra, cdec],
                                                             names=['ra', 'dec']), arcsecrad=5)
        if (index!=-1).sum()>4:
            wc_cand, index = refine_transfo(wc_cand, refcat, cat, astropy=astropy, prefix=prefix)
        matched1 = (index!=-1).sum()
        return wc_cand, matched1

class FittedCoordinateTransformation(object):
    ''' Affine transformation between two coordinate systems
    '''
    def __init__(self):
        ''' Initialize with identity transform
        '''
        self.affine_transform = np.hstack([np.eye(2), [[0],[0]]])

    def __call__(self, x, y):
        ''' Apply the transformation to the provided coordinates
        '''
        xt, yt = np.dot(self.affine_transform, np.array([x, y, np.ones(len(x))]))
        return xt, yt

    def inverse(self, x, y):
        M = np.linalg.inv(np.vstack([self.affine_transform, [0, 0,1]]))
        X, Y, un = np.dot(M, np.array([x,
                                       y,
                                       np.ones(len(x))]))
        return X, Y
    
    def fit(self, drills, pix):
        """ Fit 4 holes to get a transformation
        """    
        X, Y = drills['x'].squeeze(), drills['y'].squeeze()
        x, y = pix['x'], pix['y']
        A = np.array([x,y, np.ones(len(x))])
        self.affine_transform = np.array([np.linalg.solve(np.dot(A, A.T), np.dot(A, X)),
                                          np.linalg.solve(np.dot(A, A.T), np.dot(A, Y))])

class AlignmentFinder(object):
    def __init__(self, drills, match_radius=0.5, precision=25, heuristic_resolution=8):
        self.drills = drills
        self.radius = match_radius
        self.precision = precision
        self.heuristic_resolution = heuristic_resolution

        # Prepare a hash table to find candidate transformations
        quadrangles = get_quad_combinations(drills, 8)
        hashes, self.sorted_quadrangles = get_astrometric_hash(self.drills, quadrangles, precision=self.precision, mirror=True)
        hash_t = [tuple(a) for a in hashes.T]
        self.quadrangle_map = dict(zip(hash_t, np.arange(quadrangles.shape[0])))

        # Prepare a hash table to match holes with the drill coordinates
        self.assoc = NearestNeighAssoc(first=[self.drills['x'], self.drills['y']], radius=self.radius)

    def candidate_to_transform(self, holes, candidate):
        q1, q2 = candidate[1], self.sorted_quadrangles[candidate[2]]
        transform = FittedCoordinateTransformation()
        transform.fit(self.drills[np.array(q2)], holes[np.array(q1)])
        tx, ty = transform(holes['x'], holes['y'])
        index = self.assoc.match(tx, ty, metric=euclidian)
        matched = index != -1
        return transform, index
    
    def eval_transform(self, holes, candidate):
        transform, index = self.candidate_to_transform(holes, candidate)
        return np.sum(index != -1)

    def refine_transform(self, holes, candidate):
        transform, index = self.candidate_to_transform(holes, candidate)
        matched = index != -1
        transform.fit(self.drills[index[matched]], holes[matched])
        tx, ty = transform(holes['x'], holes['y'])
        index = self.assoc.match(tx, ty, metric=euclidian)
        return transform, index
    
    def find_transform(self, holes):
        candidates = quad_search(holes, self.quadrangle_map, precision=self.precision)        
        nmatches = [self.eval_transform(holes, candidate) for candidate in candidates]
        if len(nmatches) > 0:
            best = np.argmax(nmatches)
            return self.refine_transform(holes, candidates[best])
        else:
            return None, None
        
    def plot_transform(self, holes, transform, index):
        tx, ty = transform(holes['x'], holes['y'])
        matched = index != -1
        plt.plot(self.drills['x'], self.drills['y'], 'o')
        plt.plot(tx[matched], ty[matched], '+')
        plt.plot(tx[~matched], ty[~matched], 'rx')

    def show_transform(self, holes, transform, index, frame):
        dx, dy = transform.inverse(self.drills['x'], self.drills['y'])
        matched = index != -1
        try:
            for x, y in zip(dx[index[matched]], dy[index[matched]]):
                cv2.circle(frame, (round(x), round(y)), 2, (0, 255, 0), -1)
        except Exception as e:
            import pdb
            pdb.set_trace()
