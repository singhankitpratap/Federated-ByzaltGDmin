# Data Generation
Run 
```sh
python data.py
```
This will save 
 - `scalars.json`: Contains value assigned to $`n`$,$`m`$,$`\tilde{m}`$,$`q`$,$`r`$,$`L`$,$`\sigma_{1}^\ast`$,$`\mu`$,$`\kappa`$
 - `data_nodes.pth`: $`data\_node[\ell]`$ contains generated data $`(A_k)_\ell`$,$`(y_k)_\ell`$ for $`\ell=0:L-1`$
 - `U_star.pth`: $`U^\ast`$
 - `data_node_key.pth`: Contains federated data for node `key`$`=0:L-1`$

Values $`n`$,$`m`$,$`q`$,$`r`$, and $`L=`$`Number of federated clients` can be tuned accordingly for the experiment.
