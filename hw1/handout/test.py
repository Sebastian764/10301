
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.leftNode = None
        self.rightNode = None

def insert_node(root, key):
    if root is None:
        return TreeNode(key)
    else:
        if root.key < key:
            root.rightNode = insert_node(root.rightNode, key)
        else:
            root.leftNode = insert_node(root.leftNode, key)
    return root

def create_test_tree():
    keys = [20, 10, 30, 5, 15, 25, 35, 2, 7, 12, 18, 22, 28, 32, 38]
    root = None
    for key in keys:
        root = insert_node(root, key)
    return root

# Function to search for a value in the tree using DFS
def find_val(node, key):
    if node is None:
        return None
    if node.key == key:
        return node
    else:
        result = find_val(node.rightNode, key)
        if result is None:
            result = find_val(node.leftNode, key)
        return result

# Create a test tree and test the find_val function
test_tree = create_test_tree()
search_key = 15
found_node = find_val(test_tree, search_key)

# Check if the function works correctly
if found_node:
    print(f"Node with key {search_key} found.")
else:
    print(f"Node with key {search_key} not found.")