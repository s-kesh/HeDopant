import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


def read_arrow_traj(path):
    """
    Reads an Arrow IPC stream created by ArrowIO (C++ code).
    Converts nested list column 'y' → NumPy 2D array.
    Returns a pandas DataFrame with columns:
        step, rows, cols, data
    """
    df_rows = []

    with pa.memory_map(path, "r") as source:
        reader = ipc.RecordBatchStreamReader(source)

        for batch in reader:
            step_col = batch.column(0)
            y_col = batch.column(1)  # nested list of list of float64

            for i in range(batch.num_rows):
                step = step_col[i].as_py()

                # Extract nested lists
                matrix_list = y_col[i].as_py()  # list of lists
                arr = np.array(matrix_list, dtype=float)

                df_rows.append(
                    {
                        "step": step,
                        "rows": arr.shape[0],
                        "cols": arr.shape[1],
                        "data": arr,
                    }
                )

    return pd.DataFrame(df_rows)
