import numpy as np
from scipy.optimize import fsolve

e = 1.602176634e-19   # elementary charge, C
h = 6.626e-34          # Planck constant, J*s

def ECQ(CJ, CQ, CD):
    return (e**2 / 2) * ((CD + CJ) / (CD * CQ + CD * CJ + CQ * CJ))


def ECD(CJ, CQ, CD):
    return (e**2 / 2) * ((CQ + CJ) / (CD * CQ + CD * CJ + CQ * CJ))

def EJD(ratioD, CJ, CQ, CD):
    return ratioD * ECD(CJ, CQ, CD)


def EJQ(ratioQ, CJ, CQ, CD):
    return ratioQ * ECQ(CJ, CQ, CD)

def omegaQ(ratioQ,CJ, CQ, CD):
    return np.sqrt(8 * ECQ(CJ, CQ, CD) * EJQ(ratioQ, CJ, CQ, CD)) - ECQ(CJ, CQ, CD)


def omegaD(ratioD,CJ, CQ, CD):
    return np.sqrt(8 * ECD(CJ, CQ, CD) * EJD(ratioD,CJ, CQ, CD)) - ECD(CJ, CQ, CD)

def J(ratioQ, ratioD, CJ, CQ, CD):
    return (
       ( -4
        * e**2)
        * (CJ / (CD * CQ + CD * CJ + CQ * CJ))
        * ( EJQ(ratioQ, CJ, CQ, CD) * EJD(ratioD, CJ, CQ, CD)/ (32*ECQ(CJ, CQ, CD)*32*ECD(CJ, CQ, CD))) ** (1 / 4)
    )


def solve_EJ_ratio(omega,eta):
    E_C = -eta
    E_J = (omega + E_C)**2 / 8/ E_C

    print(f'Target EJ: {E_J/1e9} GHz')
    print(f'Target EJ/EC: {E_J/E_C}')

    return E_J/E_C

def solve_EC_from_ratio(omega,ratio):
    E_C = omega / (np.sqrt(8*ratio)-1)

    print(f'Target EC: {E_C/1e6} MHz')
    return E_C

def solve_J(targetOmegaQ_Hz, targetOmegaD_Hz, targetEtaQ_Hz, targetRatioD, targetJ_Hz): 
    targetOmegaQh = targetOmegaQ_Hz * h # Qubit frequency
    targetOmegaDh = targetOmegaD_Hz * h # Dissipator Frequency
    targetJh =  targetJ_Hz * h # Target coupliing
    ratioQ = solve_EJ_ratio(targetOmegaQ_Hz, targetEtaQ_Hz)
    ratioD = targetRatioD
    def equations(x_fF,targetOmegaQ,targetOmegaD,targetJ,ratioQ,ratioD):

        CJ, CQ, CD = [v * 1e-15 for v in x_fF]
        return [
            (omegaQ(ratioQ,CJ, CQ, CD) - targetOmegaQ) / targetOmegaQ,
            (omegaD(ratioD,CJ, CQ, CD) - targetOmegaD) / targetOmegaD,
            (J(ratioQ, ratioD, CJ, CQ, CD)- targetJ) / targetJ,
        ]

    # qubit-diss capacitance, qubit capcaitance, dissipator capacitance inital guesses
    x0_fF = [1.5, 97, 75]
    sol_fF = fsolve(equations, x0_fF,args=(targetOmegaQh,targetOmegaDh,targetJh,ratioQ,ratioD), full_output=False)
    CJsol, CQsol, CDsol = [v * 1e-15 for v in sol_fF]
    print(f"CJ = {CJsol / 1e-15} fF")
    print(f"CQ = {CQsol / 1e-15} fF")
    print(f"CD = {CDsol / 1e-15} fF")
    print(f"Verify omegaQ = {omegaQ(ratioQ,CJsol, CQsol, CDsol) / h / 1e9} GHz (target: {targetOmegaQ_Hz/1e9})")
    print(f"Verify omegaD = {omegaD(ratioD,CJsol, CQsol, CDsol) / h / 1e9} GHz (target: {targetOmegaD_Hz/1e9})")
    print(f"Verify J      = {J(ratioQ,ratioD,CJsol, CQsol, CDsol) / h / 1e6} MHz (target: {targetJ_Hz/1e6})")
    print(f"Verify EJQ/ECQ = {EJQ(ratioQ,CJsol, CQsol, CDsol) / ECQ(CJsol, CQsol, CDsol)} (target: {ratioQ})")  # NOTE: the original notebook's Print statement says "target: 50" here even though ratioQ = 70 -- likely a leftover/typo in the original, kept as-is for fidelity
    print(f"Verify EJD/ECD = {EJD(ratioD,CJsol, CQsol, CDsol) / ECD(CJsol, CQsol, CDsol)} (target: {ratioD})")


    return 