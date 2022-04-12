import numpy as np

# overall constraint loss - WIP
def overall_loss (output, ground_truth, constraint_variables):

    constraint_loss = 0

    sensitivity_matrix = [1.0, 0.0, 0.0, 0.0, 0.0] # will come from config file.

    for time in range(output.shape[0]):

        constraint_loss = sensitivity_matrix[0] * precip_pos(output) + sensitivity_matrix[1] * moisture_budget(output, constraint_variables) + sensitivity_matrix[2] * mass_budget(output, ground_truth) + sensitivity_matrix[3] * climate_energy_budget(output, ground_truth, constraint_variables)

    return constraint_loss

# setting any negative precip value to positive, used after each forward pass
def precip_pos(output):

    # shape of model output [time, resolution, fields ]

    # fields are [surf temp, surf pressure, precip]

    precip = np.array(output[:, :, 2]) # update index from 0 to the correct number

    precip[precip<0] = 0 # set any negative values to zero

    output[:, :, 2] = precip # update original precip value to reflect updated array

    return output

# check if balance between evap and precip is balanced, at every point, after each forward pass
def moisture_budget(output, constraint_variables):

    # shape of output [time, resolution, fields ]

    # shape of constraint_variables is [time, resolution, fields]

    # calculate difference across all resolution across same time

    loss_mb = np.zeros((output.shape[0], output.shape[1])) # setting a zero array with shape (time, resolution)

    for time in range(0, output.shape[0]):

        mb = output[time, :, 0] - constraint_variables[time, :, 0] # assuming the relevant indices are zero

        loss_mb[time, :] = loss_mb[time, :] + mb

    return loss_mb


def mass_budget(output, ground_truth):

    # summation across all spatial points at time t from the model must be equal to the ground truth for surface_pres

    loss_mass = np.zeros((output.shape[0], output.shape[1]))  # setting a zero array with shape (time, resolution)

    for time in range(0, output.shape[0]):

        mass_diff = output[time, :, 1] - ground_truth[time, :, 1]  # assuming the relevant indices is 1

        loss_mass[time, :] = loss_mass[time, :] + mass_diff

    return loss_mass

def trop_energy_budget():

    # too many terms. Needs further simplication

    return None

def climate_energy_budget(output, ground_truth, constraint_variables):

    # sum up upto a year or more?

    lambda_feedback = 0.97 # config['lambda'] # get the updated value from UVic, should be in config file as input and be read from that

    loss_climate_energy = np.zeros(output.shape[0], output.shape[1])

    for time in range(0, output.shape[0]):

       energy_budget = constraint_variables[time, :, 1] - lambda_feedback * (output[time, :, 2] - ground_truth[time, :, 2])

        loss_climate_energy[time, :] = loss_climate_energy[time, :] + energy_budget

    return loss_climate_energy