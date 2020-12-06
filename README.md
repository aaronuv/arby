# Arby

[Model Order Reduction (MOR)](https://en.wikipedia.org/wiki/Model_order_reduction)
is a technique for reducing the computational complexity of mathematical models in
numerical simulations.

Arby is a fully data-driven Python module to construct reduced bases,
empirical interpolants and surrogate models from training data.

# Install

    pip install arby

# Quick Usage

Suppose we want to build a surrogate model for a family of real functions $f_\lambda(x)$
parametrized by a real number $\lambda\in[\lambda_{min},\lambda_{max}]$ and $x\in[a,b]$.
We have discretizations of both domains in, say, 101 and 1001 points respectively,
```
lambda_params = np.linspace(lambda_min, lambda_max, 101)
x_samples = np.linspace(a, b, 1001)
```
The next step is to build a training set of functions associated to the discretizations.

```
training_data = [f(lambda, x_samples) for lambda in lambda_params]
```
This is an array of shape $(101,1001)$.

Then we can build a surrogate model with `arby` using:

    from arby import ReducedOrderModeling as ROM
    f_model = ROM(training_space=training_data,
                  physical_interval=x_samples,
                  parameter_interval=lambda_params)
    
With our `f_model` we can get function samples for any parameter $\lambda$ in the
interval $\lambda\in[\lambda_{min},\lambda_{max}]$.

    new_param = 0.554
    f_model_new_param = f_model.surrogate(new_param)
    plt.plot(x_samples, model_new_param)
    plt.show()

# License

MIT

***

(c) 2020 Aaron Villanueva
