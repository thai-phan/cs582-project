from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from src.evaluation.metrics import checked


@dataclass
class CalibratedModel:
    base: object
    slope: float
    intercept: float

    @property
    def name(self): return self.base.name+'_calibrated'

    @classmethod
    def fit(cls,fitted,Xval,yval):
        y,p=checked(yval,fitted.predict_proba(Xval)[:,1])
        z=logit(np.clip(p,1e-6,1-1e-6))
        def objective(params):
            a,b=params; score=a*z+b
            loss=np.mean(np.logaddexp(0,score)-y*score)+1e-6*a*a
            residual=expit(score)-y
            grad=np.array([np.mean(residual*z)+2e-6*a,np.mean(residual)])
            return loss,grad
        result=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(1e-6,20),(-20,20)])
        if not result.success: raise RuntimeError(f'calibration failed: {result.message}')
        return cls(fitted,float(result.x[0]),float(result.x[1]))

    def transform(self,p):
        return expit(self.slope*logit(np.clip(p,1e-6,1-1e-6))+self.intercept)

    def predict_proba(self,X):
        p=self.transform(self.base.predict_proba(X)[:,1])
        return np.column_stack([1-p,p])
