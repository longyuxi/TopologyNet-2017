## This file is part of PATH, which is part of OSPREY 3.0
##
## OSPREY Protein Redesign Software Version 3.0
## Copyright (C) 2001-2023 Bruce Donald Lab, Duke University
##
## OSPREY is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License version 2
## as published by the Free Software Foundation.
##
## You should have received a copy of the GNU General Public License
## along with OSPREY.  If not, see <http://www.gnu.org/licenses/>.
##
## OSPREY relies on grants for its development, and since visibility
## in the scientific literature is essential for our success, we
## ask that users of OSPREY cite our papers. See the CITING_OSPREY
## document in this distribution for more information.
##
## Contact Info:
##    Bruce Donald
##    Duke University
##    Department of Computer Science
##    Levine Science Research Center (LSRC)
##    Durham
##    NC 27708-0129
##    USA
##    e-mail: www.cs.duke.edu/brd/
##
## <signature of Bruce Donald>, Mar 1, 2023
## Bruce Donald, Professor of Computer Science

from calculate_gradients import DB, KEY_PREFIX
from tqdm import tqdm
import pandas as pd

correct_value = 6.0

database_keys = DB.keys(KEY_PREFIX + '*')
database_keys.sort(key=lambda x: int(x.split('_')[-1]))

df = pd.DataFrame()
for key in tqdm(database_keys):
    info = DB.hgetall(key)

    for k, v in info.items():
        # Set the row to the key
        df.loc[key, k] = v

    df.loc[key, 'diff'] = float(info['y_hat']) - correct_value


df.to_csv(f'{KEY_PREFIX}_results.csv')