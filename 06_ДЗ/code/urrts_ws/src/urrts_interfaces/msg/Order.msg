# Order: carry an item from a pickup station to a drop station.
string id
string pickup            # station name, see config/warehouse.yaml
string drop
float64 reward           # value of the order before time discount
builtin_interfaces/Time created
