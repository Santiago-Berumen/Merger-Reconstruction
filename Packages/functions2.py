import pandas as pd
import precession

def generate_initial_population(N, priors):
    # Sample N from the prior
    sample = priors.sample(N)

    # Create dataframe
    df = pd.DataFrame.from_dict(sample)

    # Add a label setting this as the initial population (N_merge = 0)
    df['N_merge'] = 0

    # For the initial population, set the recoil kick to 0
    df['kick'] = 0.0

    # For the initial population, set the parent ids to [-1, -1]
    df['parent_ids'] = [(-1, -1)] * len(df)

    return df


def get_remnant_properties(mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12,
                           package="precession", return_kick_vector=False):

    if package == "precession":
        # Define the mass ratio
        q = mass_2 / mass_1

        # precession expects the primary to be the heavier black hole (0 <= q <= 1).
        # It does not check this itself, so a swapped pair returns wrong numbers
        # silently. Run the inputs through reorder_masses first.
        if q > 1:
            raise ValueError(
                f"mass_2 ({mass_2}) is larger than mass_1 ({mass_1}), giving q = {q} > 1. "
                "Call reorder_masses() before get_remnant_properties()."
            )

        # Compute remnant properties using precession package
        # precession.remnantmass returns fraction of (m1+m2)
        final_mass = precession.remnantmass(tilt_1, tilt_2, q, a_1, a_2)[0] * (mass_1 + mass_2)
        final_spin = precession.remnantspin(tilt_1, tilt_2, phi_12, q, a_1, a_2)[0]
        recoil_kick = precession.remnantkick(tilt_1, tilt_2, phi_12, q, a_1, a_2, kms=True, full_output=True)[0]

        # Separate kick magnitude and vector
        # recoil_kick is of the form [|v|, v_x, v_y, v_z]
        recoil_kick_magnitude = recoil_kick[0]
        recoil_kick_vector = recoil_kick[1:]

    else:
        raise ValueError(f"Unsupported package: {package!r}. Only 'precession' is implemented.")

    if return_kick_vector:
        return final_mass, final_spin, recoil_kick_vector
    else:
        return final_mass, final_spin, recoil_kick_magnitude


def reorder_masses(mass_1, mass_2, a_1, a_2, tilt_1, tilt_2):
    # Ensure mass_1 is the primary (largest)
    if mass_2 > mass_1:
        # Swap all parameters accordingly
        mass_1, mass_2 = mass_2, mass_1
        a_1, a_2 = a_2, a_1
        tilt_1, tilt_2 = tilt_2, tilt_1

    return mass_1, mass_2, a_1, a_2, tilt_1, tilt_2


def merge_two_random(population, priors):
    # sample 2 random rows (keep their original indices so the parents stay traceable)
    random_bhs = population.sample(n=2)
    parent_ids = (random_bhs.index[0], random_bhs.index[1])

    # create a new dataframe without those sampled rows
    new_pop = population.drop(random_bhs.index)

    # extract properties of the two random black holes
    # each row is a single black hole, so its spin magnitude/tilt live in
    # the 'a_1'/'tilt_1' columns; the secondary's tilt is drawn from its own row
    mass_1 = float(random_bhs.iloc[0]['mass_1'])
    a_1 = float(random_bhs.iloc[0]['a_1'])
    tilt_1 = float(random_bhs.iloc[0]['tilt_1'])
    phi_12 = float(random_bhs.iloc[0]['phi_12'])
    mass_2 = float(random_bhs.iloc[1]['mass_1'])
    a_2 = float(random_bhs.iloc[1]['a_1'])
    tilt_2 = float(random_bhs.iloc[1]['tilt_2'])

    # reorder by larger mass
    mass_1, mass_2, a_1, a_2, tilt_1, tilt_2 = reorder_masses(
        mass_1, mass_2, a_1, a_2, tilt_1, tilt_2
    )

    # get the remnant properties
    final_mass, final_spin, final_kick = get_remnant_properties(
        mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12
    )

    # sample some new values from prior to use for tilts, phase, etc...
    new_params = priors.sample(1)

    # add the remnant to the new population
    new_row = {
        "mass_1": final_mass,
        "a_1": final_spin,
    
        # remnant-independent parameters
        "tilt_1": new_params["tilt_1"][0],
        "tilt_2": new_params["tilt_2"][0],
        "phi_12": new_params["phi_12"][0],
        "phi_jl": new_params["phi_jl"][0],
        "phase": new_params["phase"][0],
        "theta_jn": new_params["theta_jn"][0],
    
        # set the N_merge and kick velocity
        "N_merge": int(random_bhs.iloc[0]['N_merge'] + random_bhs.iloc[1]['N_merge'] + 1),
        "kick": final_kick,

        # save the parent ids for tracking
        "parent_ids": parent_ids
    }

    # give the remnant a fresh index label so the parent ids stay unambiguous
    idx = int(population.index.max()) + 1
    new_pop.loc[idx] = new_row
    new_pop = new_pop.sort_index()

    merger_row = {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "a_1": a_1,
        "a_2": a_2,
        "tilt_1": tilt_1,
        "tilt_2": tilt_2,
        "Ng": int(random_bhs.iloc[0]['N_merge'] + random_bhs.iloc[1]['N_merge'] + 1),
        "m_f": final_mass,
        "a_f": final_spin,
        "v_k": final_kick,
        "parent_ids": parent_ids
    }

    return new_pop, merger_row


def ng1g_merger(population, priors, N_to_merge):
    if N_to_merge == 0:
        return merge_two_random(population, priors)
    else:
        mask = population["N_merge"] == N_to_merge
        nth_gens = population[mask]

        # Check that we have the right size
        if len(nth_gens) == 0 :
            raise ValueError(f"No black holes with N_merge = {N_to_merge} in the population.")
        elif len(nth_gens) > 1:
            print(f"More than one black hole with N_merge = {N_to_merge} in the population. Picking a random one.")
            nth_gens = nth_gens.sample(n=1)

        # Merge it with an N_merge = 0
        mask = population["N_merge"] == 0
        first_gens = population[mask]
        
        if len(first_gens) == 0 :
            raise ValueError(f"No first-generation black holes in the population.")

        # sample a first generation black hole to merge with
        random_first_gen = first_gens.sample(n=1)

        # keep the original indices so the parents stay traceable
        parent_ids = (nth_gens.index[0], random_first_gen.index[0])

        # create a new dataframe without the two merging black holes.
        # this has to come off the full population -- dropping them from
        # first_gens instead would silently discard every other generation
        new_pop = population.drop(index=list(parent_ids))

        # extract properties of the two black holes
        # each row is a single black hole, so its spin magnitude/tilt live in
        # the 'a_1'/'tilt_1' columns; the secondary's tilt is drawn from its own row
        mass_1 = float(nth_gens.iloc[0]['mass_1'])
        a_1 = float(nth_gens.iloc[0]['a_1'])
        tilt_1 = float(nth_gens.iloc[0]['tilt_1'])
        phi_12 = float(nth_gens.iloc[0]['phi_12'])
        mass_2 = float(random_first_gen.iloc[0]['mass_1'])
        a_2 = float(random_first_gen.iloc[0]['a_1'])
        tilt_2 = float(random_first_gen.iloc[0]['tilt_2'])

        # reorder by larger mass
        mass_1, mass_2, a_1, a_2, tilt_1, tilt_2 = reorder_masses(
            mass_1, mass_2, a_1, a_2, tilt_1, tilt_2
        )
        
        # get the remnant properties
        final_mass, final_spin, final_kick = get_remnant_properties(
            mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12
        )
        
        # sample some new values from prior to use for tilts, phase, etc...
        new_params = priors.sample(1)
        
        # add the remnant to the new population
        new_row = {
            "mass_1": final_mass,
            "a_1": final_spin,
        
            # remnant-independent parameters
            "tilt_1": new_params["tilt_1"][0],
            "tilt_2": new_params["tilt_2"][0],
            "phi_12": new_params["phi_12"][0],
            "phi_jl": new_params["phi_jl"][0],
            "phase": new_params["phase"][0],
            "theta_jn": new_params["theta_jn"][0],
        
            # set the N_merge and kick velocity
            "N_merge": int(nth_gens.iloc[0]['N_merge'] + random_first_gen.iloc[0]['N_merge'] + 1),
            "kick": final_kick,

            # save the parent ids for tracking
            "parent_ids": parent_ids
        }

        # give the remnant a fresh index label so the parent ids stay unambiguous
        idx = int(population.index.max()) + 1
        new_pop.loc[idx] = new_row
        new_pop = new_pop.sort_index()

        merger_row = {
            "mass_1": mass_1,
            "mass_2": mass_2,
            "a_1": a_1,
            "a_2": a_2,
            "tilt_1": tilt_1,
            "tilt_2": tilt_2,
            "Ng": int(nth_gens.iloc[0]['N_merge'] + random_first_gen.iloc[0]['N_merge'] + 1),
            "m_f": final_mass,
            "a_f": final_spin,
            "v_k": final_kick,
            "parent_ids": parent_ids
        }

        return new_pop, merger_row

def pairing_function(mass_1, mass_2, beta1, beta2):
    q = min(mass_1, mass_2) / max(mass_1, mass_2)  # always <= 1
    M_tot = mass_1 + mass_2
    return q**beta1 * M_tot**beta2
        
def n1gn2g_merger(population, priors, N_to_merge_1, N_to_merge_2, beta1, beta2, idx=None):
    # pick the first black hole from the population with N_merge = N_to_merge_1
    mask1 = population["N_merge"] == N_to_merge_1
    population1 = population[mask1]

    if len(population1) == 0:
        raise ValueError(f"No black holes with N_merge = {N_to_merge_1} to merge.")

    # sample the first black hole
    random_bh1 = population1.sample(n=1)

    # extract properties of the two black holes
    mass_1 = float(random_bh1.iloc[0]['mass_1'])
    a_1 = float(random_bh1.iloc[0]['a_1'])
    tilt_1 = float(random_bh1.iloc[0]['tilt_1'])
    phi_12 = float(random_bh1.iloc[0]['phi_12'])

    # pick the second black hole from the population with N_merge = N_to_merge_2
    # need to make sure that if N_to_merge_1 == N_to_merge_2, we don't pick the same black hole
    temp_population = population.drop(random_bh1.index)
    mask2 = temp_population["N_merge"] == N_to_merge_2
    population2 = temp_population[mask2]

    if len(population2) == 0:
        raise ValueError(f"No black holes with N_merge = {N_to_merge_2} to merge.")


    #integrate over all remaining BHs and assign them weights according to the pairing function
    weights = population2['mass_1'].apply(lambda m2: pairing_function(mass_1, m2, beta1, beta2))
    #print(weights.sort_values(ascending=False).head(10))
    #sample a random black hole with weights applies
    random_bh2 = population2.sample(n=1, weights=weights)


    # create a new dataframe without the sampled rows
    # we already removed the first black hole from temp_population
    new_pop = temp_population.drop(random_bh2.index)

    # reset the indices
    #random_bh1 = random_bh1.reset_index(drop=True)
    #random_bh2 = random_bh2.reset_index(drop=True)
    #new_pop = new_pop.reset_index(drop=True)

    
    mass_2 = float(random_bh2.iloc[0]['mass_1'])
    a_2 = float(random_bh2.iloc[0]['a_1'])
    tilt_2 = float(random_bh2.iloc[0]['tilt_2'])

    # reorder by larger mass
    mass_1, mass_2, a_1, a_2, tilt_1, tilt_2 = reorder_masses(
        mass_1, mass_2, a_1, a_2, tilt_1, tilt_2
    )
    
    # get the remnant properties
    final_mass, final_spin, final_kick = get_remnant_properties(
        mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12
    )

    # sample some new values from prior to use for tilts, phase, etc...
    new_params = priors.sample(1)
    
    # add the remnant to the new population
    new_row = {
        "mass_1": final_mass,
        "a_1": final_spin,
    
        # remnant-independent parameters
        "tilt_1": new_params["tilt_1"][0],
        "tilt_2": new_params["tilt_2"][0],
        "phi_12": new_params["phi_12"][0],
        "phi_jl": new_params["phi_jl"][0],
        "phase": new_params["phase"][0],
        "theta_jn": new_params["theta_jn"][0],
    
        # set the N_merge and kick velocity
        "N_merge": int(random_bh1.iloc[0]['N_merge'] + random_bh2.iloc[0]['N_merge'] + 1),
        "kick": final_kick,

        # save the parent ids for tracking
        "parent_ids": (random_bh1.index[0], random_bh2.index[0])
    }

    #new_pop.loc[len(new_pop)] = new_row
    if idx is None:
        # fresh label -- reusing a parent's label would make parent_ids ambiguous
        idx = int(population.index.max()) + 1
    new_pop.loc[idx] = new_row
    new_pop = new_pop.sort_index()

    merger_row = {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "a_1": a_1,
        "a_2": a_2,
        "tilt_1": tilt_1,
        "tilt_2": tilt_2,
        "Ng": int(random_bh1.iloc[0]['N_merge'] + random_bh2.iloc[0]['N_merge'] + 1),
        "m_f": final_mass,
        "a_f": final_spin,
        "v_k": final_kick,
        "parent_ids": (random_bh1.index[0], random_bh2.index[0])
    }

    return new_pop, merger_row


def create_ng1g_chain(priors, N_initial, N_merge_max=None):
    populations = {}

    # Create the initial population
    populations[0] = generate_initial_population(N_initial, priors)

    # Merge once to create an N_merge = 1 black hole
    populations[1], _ = merge_two_random(populations[0], priors)

    if N_merge_max is None:
        N_merge_max = N_initial - 1
    if N_merge_max > (N_initial - 1):
        N_merge_max = N_initial - 1
        print(f"Not enough black holes to merge until N_merge_max. Setting N_merge_max = {N_merge_max}.")

    for N_to_merge in range(1, N_merge_max):
        populations[N_to_merge+1], _ = ng1g_merger(
            populations[N_to_merge], 
            priors, 
            N_to_merge)

    return populations


def create_ng1g_chain_mergers(priors, N_initial, N_merge_max=None, verbose=False):
    # Initialize merger list
    merger_rows = []
    
    # Create the initial population
    population = generate_initial_population(N_initial, priors)

    # Merge once to create an N_merge = 1 black hole
    population, new_merger_row = merge_two_random(population, priors)
    merger_rows.append(new_merger_row)

    # Set maximum amount of mergers
    if N_merge_max is None:
        N_merge_max = N_initial - 1
    if N_merge_max > (N_initial - 1):
        N_merge_max = N_initial - 1
        print(f"Not enough black holes to merge until N_merge_max. Setting N_merge_max = {N_merge_max}.")

    # Loop over NG + 1G mergers
    for N_to_merge in range(1, N_merge_max):
        population, new_merger_row = ng1g_merger(population, priors, N_to_merge)
        merger_rows.append(new_merger_row)

    mergers = pd.DataFrame(merger_rows)

    return mergers


def create_ngng_chain(priors, N_initial, beta1, beta2):
    populations = {}

    # Create the initial population
    populations[0] = generate_initial_population(N_initial, priors)

    i = 0

    while True:
        # Find the largest N_merge value that has at least 2 entries
        unique_Ns = sorted(populations[i]["N_merge"].unique(), reverse=True)
        merge_level = None
        for N in unique_Ns:
            count = (populations[i]["N_merge"] == N).sum()
            if count >= 2:
                merge_level = N
                break
    
        # If none has 2 or more, break or regenerate
        if merge_level is None:
            print("No levels with 2 or more systems to merge. Stopping.")
            break
    
        # perform the merger at that level
        idx = N_initial + i
        populations[i+1], merger_row = n1gn2g_merger(
            populations[i], priors, merge_level, merge_level, beta1, beta2, idx=idx
        )

        i += 1

    return populations


def create_ngng_chain_mergers(priors, N_initial, v_esc, beta1, beta2, tracked_idx=None, verbose=False, return_all=False):
    # Initialize merger list
    merger_rows = []

    # Create the initial population
    population = generate_initial_population(N_initial, priors)

    i = 0

    while True:
        # find the largest N_merge value that has at least two entries
        unique_Ns = sorted(population["N_merge"].unique(), reverse=True)
        merge_level = None # this will be the highest N_merge level that has at least two entries
        for N in unique_Ns:
            count = (population["N_merge"] == N).sum()
            if count >= 2:
                merge_level = N
                break
    
        # if none has two or more, break or regenerate
        if merge_level is None:
            if verbose:
                print("No levels with 2 or more systems to merge. Stopping.")
            break
    
        # perform the merger at that level
        idx = N_initial + i
        population, merger_row = n1gn2g_merger(population, priors, merge_level, merge_level, beta1, beta2, idx=idx)

        # extract the parent's ids
        parent_ids = population.at[idx, "parent_ids"]

        if tracked_idx is None and i == 0:
            # to maximize the amount of mergers a tracked_id can get,
            # i'll track the first one that merges
            tracked_idx = parent_ids[0]

        # is this merger part of the tracked black hole's lineage?
        is_tracked = tracked_idx in parent_ids

        # set whether to return all mergers or only mergers involving
        # a tracked black hole
        if return_all or is_tracked:
            merger_rows.append(merger_row)

        if merger_row['v_k'] > v_esc:
            # the remnant was kicked out of the cluster, so it can never merge again
            population = population.drop(index=idx)

            # only the tracked black hole leaving ends the chain. an unrelated
            # remnant escaping elsewhere in the cluster must not stop the run
            if is_tracked:
                break

        elif is_tracked:
            # the remnant was retained, so follow it into the next generation
            tracked_idx = idx

        i += 1

    mergers = pd.DataFrame(merger_rows)

    return mergers