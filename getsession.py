from snowflake.snowpark.session import Session
import configparser
 
def read_config():
    """
    Read Snowflake account details from the configuration file.
   
    Returns:
        tuple: A tuple containing account, username, password, role, warehouse,
               database, and schema.
    """
    parser = configparser.ConfigParser()
    try:
        parser.read("./config.config")
        acc = parser.get("snowflake_connection", "account")
        usr = parser.get("snowflake_connection", "username")
        pwd = parser.get("snowflake_connection", "password")
        role = parser.get("snowflake_connection", "role")
        wh = parser.get("snowflake_connection", "warehouse")
        db = parser.get("snowflake_connection", "dbname")
        sch = parser.get("snowflake_connection", "schema")
       
        return acc, usr, pwd, role, wh, db, sch
    except Exception as e:
        print("Error reading config file:", e)
        return None, None, None, None, None, None, None
 
def create_session_object():
    """
    Create a Snowflake session object using the provided configuration details.
   
    Returns:
        Session: A Snowflake session object.
    """
    # Step 1: Read configuration parameters
    acc, usr, pwd, role, wh, db, sch = read_config()
    if acc is None or usr is None or pwd is None or role is None or wh is None or db is None or sch is None:
        print("Error: Configuration parameters are missing.")
        return None
 
    connection_parameters = {
        "account": acc,
        "user": usr,
        "password": pwd,
        "role": role,
        "warehouse": wh,
        "database": db,
        "schema": sch
    }
 
    try:
        # Step 2: Create Snowflake session
        session = Session.builder.configs(connection_parameters).create()
        # Step 3: Test session by executing a query
        print(session.sql('SELECT CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()').collect())
        return session
    except Exception as e:
        print("Error creating Snowflake session:", e)
        return None



 
def close_session(session):
    """
    Close the Snowflake session.
   
    Args:
        session (Session): The Snowflake session to be closed.
    """
    try:
        session.close()
        print("Session closed successfully.")
    except Exception as e:
        print("Error closing Snowflake session:", e)
 

