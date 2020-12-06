# Arby

[Model Order Reduction (MOR)](https://en.wikipedia.org/wiki/Model_order_reduction) is a technique for reducing the computational complexity of mathematical models in numerical simulations.

Arby is a fully data-driven Python module to construct reduced bases, empirical interpolants and surrogate models from training data.

# Install

    pip install arby

# Quick Usage

Suppose we want to build a surrogate model for a family of functions `f`
parametrized by parameter `a`.
Let's suppose we have 1,000 samples of `f` over the interval [0, 1] and for the
parameters `a` in {1, 2, 3}

Then to build the surrogate model:

    from arby import ReducedOrderModeling as ROM
    x_samples = np.linspace(0., 1., 1_000)
    model_surrogate = ROM(training_space=training_data,
                          physical_interval=x_samples,
                          parameter_interval=[1., 2., 3.])
    
With our `model_surrogate` we can get function samples for any parameter `a`.

    new_param = 0.5
    model_new_param = model_surrogate(new_param)
    plt.plot(x_samples, model_new_param)
    plt.show()

# License

MIT

***

(c) 2020 Aaron Villanueva
