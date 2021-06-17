## Namespaces
Namespaces in python are structures used to organize the symbolic names assigned to objects in a Python program. It has 4 types of namespaces:
* built-in
    * The built-in namespace contains name of all of python's built in objects
        * sum' is name of a built-in object of type builtin_function_or_method. 
        * The name 'sum' is reference to a memory location that contains instructions that will be executed when sum is invoked
    * These are available at all times when Python is running
* global
    * The global namespace contains any names defined at the level of the main program
* enclosing
* local

If your code refers to the name x, then Python searches for x in the following namespaces in the order shown:
* Local: If you refer to x inside a function, then the interpreter first searches for it in the innermost scope that’s local to that function.

* Enclosing: If x isn’t in the local scope but appears in a function that resides inside another function, then the interpreter searches in the enclosing function’s scope.

* Global: If neither of the above searches is fruitful, then the interpreter looks in the global scope next.

* Built-in: If it can’t find x anywhere else, then the interpreter tries the built-in scope.
