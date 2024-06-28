import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from skimage.transform import FundamentalMatrixTransform


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


class Extractor(object):
    GX = 16 // 2
    GY = 12 // 2

    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    def denormalize(self, pt):
        ret = np.dot(self.Kinv, np.array([pt[0], pt[1], 1.0]))
        # ret /= ret[2]
        print(ret)
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):
        features = cv2.goodFeaturesToTrack(
            np.mean(img, axis=2).astype(np.uint8),
            3000,
            qualityLevel=0.01,
            minDistance=3,
        )
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
        kps, des = self.orb.compute(img, kps)

        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last["des"], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last["kps"][m.trainIdx].pt
                    ret.append((kp1, kp2))

        if len(ret) > 0:
            ret = np.array(ret)
            ret[:, 0, :] = np.dot(self.Kinv, add_ones(ret[:, 0, :]).T).T[:, 0:2]
            ret[:, 1, :] = np.dot(self.Kinv, add_ones(ret[:, 1, :]).T).T[:, 0:2]

            model, inliners = ransac(
                (ret[:, 0], ret[:, 1]),
                # EssentialMatrixTransform,
                FundamentalMatrixTransform,
                min_samples=8,
                residual_threshold=1,
                max_trials=100,
            )
            ret = ret[inliners]
            s, v, d = np.linalg.svd(model.params)
            print(v)

        self.last = {"kps": kps, "des": des}
        return ret
