import numpy as np
from scipy.spatial.transform.rotation import Rotation
from scipy.optimize import minimize

class PoseEstimator(object):
    def __init__(self, world_coords, uvec, R=None, T = None):
        self.uvec = uvec
        self.world_coords = world_coords
        if (R is None):
            self.R0 = np.eye(3,dtype=np.double)
        else:
            self.R0 = R
        if (T is None):
            self.T0 = np.zeros(shape=3,dtype=np.double)
        else:
            self.T0 = T
    
    
    def solve(self,method='BFGS'):
        res =  minimize(self.objective, np.array([0,0,0,0,0,0]), args=(), method=method)
        R = Rotation.from_rotvec(res.x[:3]).as_euler('XYZ', degrees=True)
        T = res.x[3:]

        return R, T
    
    def objective(self, x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        T = x[3:]
        w = np.copy(self.world_coords)
        for i in range(w.shape[0]):
            w[i,:]  = w[i,:] - T
            w[i,:] = np.dot(R,w[i,:])
            w[i,:] = w[i,:]/np.linalg.norm(w[i,:])
            
        delta = self.uvec - w
        
        # self.weights = np.linalg.norm(delta, axis=1)**2
        
        loss = np.sum(delta**2)
        return loss
    
    
if __name__ == '__main__':
    
    # stickers = np.array([[5, 0, 3], [8.5, 0, 3], [2, 0, 3], [10, 2, 3], [10, 5, 3], [10, 8, 3], [0, 2, 3], [0, 5, 3]],dtype=np.double)
    
    # stickers = np.array([[0, 8, 3], [0, 6, 3], [0, 2, 3], [1, 0, 3], [7, 0, 3], [9, 0, 3], [10, 1, 3], [10, 9, 3]],dtype=np.double)
    stickers = np.array([[0, 2, 4],[0, 6, 4], [0, 8, 4], [10, 9, 4], [10, 1, 4], [9, 0, 4], [7, 0, 4], [1, 0, 4]],dtype=np.double)
    
    # stickers = np.array([[5, 0, 3], [8.5, 0, 3], [2, 0, 3],[10, 2, 3],[10, 5, 3]])
    # cam_pos = np.array([-12,5,3])
    cam_pos = np.array([5,4,4])
    
    cam_rot = Rotation.from_euler('XYZ',[0,0,55],degrees=True).as_matrix()
        
    cam_points = np.zeros_like(stickers)
    
    for i in range(stickers.shape[0]):
        p = stickers[i,:]
        p =p - cam_pos
        p = np.dot(cam_rot, p)
        p = p / np.linalg.norm(p)
        p[0] = p[0]+np.random.randn()*0.0032 # simulate noise to points
        p[1] = p[1]+np.random.randn()*0.0032 # simulatenoise to points
        p[2] = p[2]+np.random.randn()*0.0032 # simulate noise to points
        cam_points[i,:] = p
        
    stickers_pix = np.array([[63, 344, 1], [174, 363, 1], [804, 376, 1], [1242, 376, 1], [1316, 376, 1], [1412, 352, 1], [1754, 376,1], [1872, 363,1]]) #no rotation
    cam_points_pix = np.array([[318, 344,1], [430, 363,1], [1060, 376,1], [1498, 376,1], [1572, 376,1], [1669, 352,1], [2010, 376,1], [80, 363,1]]) # rotation 45 degree
    
    unit = np.array( [[-0.83654419,  0.54789174 , 0.00290888],
                    [-0.23404097,  0.97222238  ,0.00290888],
                    [ 0.05825802,  0.99829732  ,0.00290888],
                    [ 0.98474434 ,-0.17398314  ,0.00290888],
                    [ 0.07662354 ,-0.99705585  ,0.00290888],
                    [-0.17700347 ,-0.98420593  ,0.00290888],
                    [-0.47949173 ,-0.87754158  ,0.00290888],
                    [-0.98630392  ,0.16491242 , 0.00290888]]
                                        )
        
    pose = PoseEstimator(stickers, cam_points)
    
    print ('BFGS :', pose.solve())
    #print ('L-BFGS-B:', pose.solve(method='L-BFGS-B'))

            