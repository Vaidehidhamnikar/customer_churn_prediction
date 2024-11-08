from snowflake.snowpark.session import Session
import configparser

# load snowflake account details from config file
def read_config():
    parser = configparser.ConfigParser()
    a = parser.read("./config.config")
    acc = parser.get("snowflake_connection","account")
    usr = parser.get("snowflake_connection","username")
    pwd = parser.get("snowflake_connection","password")
    role = parser.get("snowflake_connection","role")
    wh = parser.get("snowflake_connection","warehouse")
    db = parser.get("snowflake_connection","dbname")
    sch = parser.get("snowflake_connection","schema")

    return acc,usr,pwd,role,wh,db,sch

# Create Session object
def create_session_object():
    """Create Snowflake session"""
    acc,usr,pwd,role,wh,db,sch = read_config()
    connection_parameters = {
  "account": acc,
  "user": usr,
  "password": pwd,
  "role": role,
  "warehouse": wh,
  "database": db,
  "schema": sch
   }
    session = Session.builder.configs(connection_parameters).create()
    print(session.sql('select current_warehouse(), current_database(), current_schema()').collect())
    return session

if __name__ == "__main__":
    create_session_object()