########################### DO NOT MODIFY THIS SECTION ##########################
#################################################################################
import sqlite3
from sqlite3 import Error
import csv
import os
#################################################################################

## Change to False to disable Sample
SHOW = True

############### SAMPLE CLASS AND SQL QUERY ###########################
######################################################################
class Sample():
    def sample(self):
        try:
            connection = sqlite3.connect("sample")
            connection.text_factory = str
        except Error as e:
            print("Error occurred: " + str(e))
        print('\033[32m' + "Sample: " + '\033[m')
        
        # Sample Drop table
        connection.execute("DROP TABLE IF EXISTS sample;")
        # Sample Create
        connection.execute("CREATE TABLE sample(id integer, name text);")
        # Sample Insert
        connection.execute("INSERT INTO sample VALUES (?,?)",("1","test_name"))
        connection.commit()
        # Sample Select
        cursor = connection.execute("SELECT * FROM sample;")
        print(cursor.fetchall())

######################################################################

class HW2_sql():
    ############### DO NOT MODIFY THIS SECTION ###########################
    ######################################################################
    def create_connection(self, path):
        connection = None
        try:
            connection = sqlite3.connect(path)
            connection.text_factory = str
        except Error as e:
            print("Error occurred: " + str(e))
    
        return connection

    def execute_query(self, connection, query):
        cursor = connection.cursor()
        try:
            if query == "":
                return "Query Blank"
            else:
                cursor.execute(query)
                connection.commit()
                return "Query executed successfully"
        except Error as e:
            return "Error occurred: " + str(e)
    ######################################################################
    ######################################################################

    # GTusername [0 points]
    def GTusername(self):
        gt_username = "gburdell3"
        return gt_username
    
    # Part a.i Create Tables [2 points]
    def part_ai_1(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_ai_1_sql = r"""
                            CREATE TABLE IF NOT EXISTS movies 
                            (id integer PRIMARY KEY,
                            title text,
                            score real);"""
        ######################################################################
        
        return self.execute_query(connection, part_ai_1_sql)

    def part_ai_2(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_ai_2_sql = r"""CREATE TABLE IF NOT EXISTS movie_cast
                            (
                                movie_id   INTEGER,
                                cast_id    INTEGER,
                                cast_name  TEXT,
                                birthday   TEXT,
                                popularity REAL
                            ); """
        ######################################################################
        return self.execute_query(connection, part_ai_2_sql)
    
    # Part a.ii Import Data [2 points]
    def part_aii_1(self,connection,path):
        ############### CREATE IMPORT CODE BELOW ############################
        # with open('./data/movies.csv', newline='', encoding="utf-8") as csvfile:
        #     movies_csv = csv.reader(csvfile)
        #     for row in movies_csv:
        #         query = f"""INSERT INTO movies (id,title,score) VALUES({row[0]},'{row[1]}',{row[2]});"""
        #         self.execute_query(connection, query)
        with open('./data/movies.csv', newline='', encoding="utf-8") as csvfile:
            movie_reader = csv.reader(csvfile)
            rows = list(movie_reader)
            query = f"""INSERT INTO movies (id,title,score) VALUES(?,?,?);"""
            cursor = connection.executemany(query, rows)
        ######################################################################
        
        sql = "SELECT COUNT(id) FROM movies;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]
    
    def part_aii_2(self,connection, path):
        ############### CREATE IMPORT CODE BELOW ############################
        # with open('./data/movie_cast.csv', newline='', encoding="utf-8") as csvfile:
        #     movies_csv = csv.reader(csvfile)
        #     for row in movies_csv:
        #         query = f"""INSERT INTO movie_cast (movie_id,cast_id,cast_name,birthday,popularity) VALUES({row[0]},{row[1]},'{row[2]}','{row[3]}',{row[4]});"""
        #         self.execute_query(connection, query)        
        with open('./data/movie_cast.csv', newline='', encoding="utf-8") as csvfile:
            movie_cast_reader = csv.reader(csvfile)
            rows = list(movie_cast_reader)
            query = f"""INSERT INTO movie_cast (movie_id,cast_id,cast_name,birthday,popularity) VALUES(?,?,?,?,?);"""
            cursor = connection.executemany(query, rows)
        ######################################################################
        
        sql = "SELECT COUNT(cast_id) FROM movie_cast;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]

    # Part a.iii Vertical Database Partitioning [5 points]
    def part_aiii(self,connection):
        ############### EDIT CREATE TABLE SQL STATEMENT ###################################
        part_aiii_sql = r"""CREATE TABLE IF NOT EXISTS cast_bio
                            (
                                cast_id    INTEGER,
                                cast_name  TEXT,
                                birthday   TEXT,
                                popularity REAL
                            ); """
        ######################################################################
        
        self.execute_query(connection, part_aiii_sql)
        
        ############### CREATE IMPORT CODE BELOW ############################
        part_aiii_insert_sql = r"""INSERT INTO cast_bio
                                    (cast_id,
                                    cast_name,
                                    birthday,
                                    popularity)
                                    SELECT DISTINCT cast_id,
                                        cast_name,
                                        birthday,
                                        popularity
                                    FROM movie_cast 
                                    """
        ######################################################################
        
        self.execute_query(connection, part_aiii_insert_sql)
        
        sql = "SELECT COUNT(cast_id) FROM cast_bio;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]
       

    # Part b Create Indexes [1 points]
    def part_b_1(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_b_1_sql = "CREATE INDEX movie_index ON movies(id);"
        ######################################################################
        return self.execute_query(connection, part_b_1_sql)
    
    def part_b_2(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_b_2_sql = "CREATE INDEX cast_index ON movie_cast(cast_id);"
        ######################################################################
        return self.execute_query(connection, part_b_2_sql)
    
    def part_b_3(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_b_3_sql = "CREATE INDEX cast_bio_index ON cast_bio(cast_id);"
        ######################################################################
        return self.execute_query(connection, part_b_3_sql)
    
    # Part c Calculate a Proportion [3 points]
    def part_c(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_c_sql = """SELECT printf("%.2f",CAST(SUM(CASE WHEN score > 50 and UPPER(title) LIKE '%WAR%' THEN 1 ELSE 0 END) AS FLOAT)/CAST(COUNT(*) AS FLOAT) *100) as proportion FROM movies;"""
        ######################################################################
        cursor = connection.execute(part_c_sql)
        return cursor.fetchall()[0][0]

    # Part d Find the Most Prolific Actors [4 points]
    def part_d(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_d_sql = "SELECT cast_name, SUM(CASE WHEN popularity > 10 THEN 1 ELSE 0 END) as popularity_over10 FROM movie_cast GROUP BY cast_name ORDER BY popularity_over10 DESC LIMIT 5;"
        ######################################################################
        cursor = connection.execute(part_d_sql)
        return cursor.fetchall()

    # Part e Find the Highest Scoring Movies With the Least Amount of Cast [4 points]
    def part_e(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_e_sql = r"""SELECT main.movie_title AS movie_title,
                        Printf("%.2f",main.movie_score) AS movie_score,
						 main.cast_count AS cast_count
						 FROM
(SELECT mc.movie_title AS movie_title,
                        mc.movie_score AS movie_score,
                        Count(*)       AS cast_count
                        FROM   movie_cast cc
                            INNER JOIN (SELECT id,
                                                title AS movie_title,
                                                score AS movie_score
                                        FROM   movies
                                        GROUP  BY id
                                        ORDER  BY score DESC) AS mc
                                    ON mc.id = cc.movie_id
                        GROUP  BY movie_id
                        ORDER  BY movie_score DESC,
                                cast_count,
                                movie_title
                        LIMIT  5) main; """
        ######################################################################
        cursor = connection.execute(part_e_sql)
        return cursor.fetchall()
    
    # Part f Get High Scoring Actors [4 points]
    def part_f(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_f_sql = r"""SELECT a.cast_id,
                        a.cast_name,
                        printf("%.2f", Sum(a.movie_score) / Count(a.cast_id)) AS average_score
                        FROM   (SELECT cc.*,
                                    mc.*
                                FROM   movie_cast AS cc
                                    INNER JOIN (SELECT id,
                                                        score AS movie_score
                                                FROM   movies
                                                WHERE  movie_score >= 25) AS mc
                                            ON mc.id = cc.movie_id
                                ORDER  BY cast_name) a
                        GROUP  BY a.cast_id
                        HAVING Count(a.cast_id) > 2
                        ORDER  BY average_score DESC, cast_name
                        LIMIT  10; """ 
        ######################################################################
        cursor = connection.execute(part_f_sql)
        return cursor.fetchall()

    # Part g Creating Views [6 points]
    def part_g(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_g_sql = r"""
                        CREATE VIEW good_collaboration 
                        AS 
                        SELECT cast_member_id1,
                            cast_member_id2,
                            Count(*) AS movie_count,
                            Printf("%.2f", Sum(movie_one.movie_score) / Count(*)) AS
                            average_movie_score
                        FROM   (SELECT a.movie_id,
                                    a.cast_id AS cast_member_id1,
                                    a.cast_name,
                                    Printf("%.2f", a.movie_score) AS movie_score
                                FROM   (SELECT cc.*,
                                            mc.*
                                        FROM   movie_cast AS cc
                                            INNER JOIN (SELECT id,
                                                        score AS movie_score
                                                        FROM   movies) AS mc
                                                    ON mc.id = cc.movie_id
                                        ORDER  BY cast_name) a
                                ORDER  BY cast_id) movie_one
                            INNER JOIN (SELECT a.movie_id,
                                                a.cast_id AS cast_member_id2,
                                                a.cast_name
                                        FROM   (SELECT cc.*,
                                                        mc.*
                                                FROM   movie_cast AS cc
                                                        INNER JOIN (SELECT id,
                                                                    score AS movie_score
                                                                    FROM   movies) AS mc
                                                                ON mc.id = cc.movie_id
                                                ORDER  BY cast_name) a
                                        ORDER  BY cast_id) movie_two
                                    ON movie_one.movie_id = movie_two.movie_id
                        WHERE  movie_one.cast_member_id1 < movie_two.cast_member_id2
                        GROUP  BY cast_member_id1,
                                cast_member_id2
                        HAVING (Sum(movie_one.movie_score) / Count(*) > 40 AND movie_count >= 3)
                        ORDER  BY cast_member_id1,cast_member_id2;"""
        ######################################################################
        return self.execute_query(connection, part_g_sql)
    
    def part_gi(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_g_i_sql = r"""SELECT DISTINCT ci.cast_id,
                            cc.cast_name,
                            ci.collaboration_score
                            FROM   (SELECT cast_member_id1 AS cast_id,
                                        Printf("%.2f", Sum(average_movie_score) / Count(cast_member_id1)) AS collaboration_score
                                    FROM   good_collaboration
                                    GROUP  BY cast_member_id1
                                    UNION
                                    SELECT cast_member_id2 AS cast_id,
                                        Printf("%.2f", Sum(average_movie_score) / Count(cast_member_id2)) AS collaboration_score
                                    FROM   good_collaboration
                                    GROUP  BY cast_member_id2) ci
                                INNER JOIN movie_cast cc
                                        ON ci.cast_id = cc.cast_id
                            ORDER  BY ci.collaboration_score DESC, cast_name
                            LIMIT  5; """
        ######################################################################
        cursor = connection.execute(part_g_i_sql)
        return cursor.fetchall()
    
    # Part h FTS [4 points]
    def part_h(self,connection,path):
        ############### EDIT SQL STATEMENT ###################################
        part_h_sql = """CREATE VIRTUAL TABLE movie_overview USING fts3(id,overview);"""
        ######################################################################
        connection.execute(part_h_sql)
        ############### CREATE IMPORT CODE BELOW ############################
        with open('./data/movie_overview.csv', newline='', encoding="utf-8-sig") as csvfile:
            movie_overview_reader = csv.reader(csvfile)
            rows = list(movie_overview_reader)
            query = f"""INSERT INTO movie_overview (id,overview) VALUES(?,?);"""
            cursor = connection.executemany(query, rows)
        ######################################################################
        sql = "SELECT COUNT(*) FROM movie_overview;"
        cursor = connection.execute(sql)
        return cursor.fetchall()[0][0]
        
    def part_hi(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_hi_sql = r"""SELECT COUNT(*) FROM movie_overview WHERE movie_overview MATCH 'FIGHT OR Fight OR fight OR fight.';"""
        ######################################################################
        cursor = connection.execute(part_hi_sql)
        return cursor.fetchall()[0][0]
    
    def part_hii(self,connection):
        ############### EDIT SQL STATEMENT ###################################
        part_hii_sql = r"""SELECT COUNT(*) FROM movie_overview WHERE movie_overview MATCH 'space NEAR/5 program';"""
        ######################################################################
        cursor = connection.execute(part_hii_sql)
        return cursor.fetchall()[0][0]


if __name__ == "__main__":
    
    ########################### DO NOT MODIFY THIS SECTION ##########################
    #################################################################################
    if SHOW == True:
        sample = Sample()
        sample.sample()

    print('\033[32m' + "Q2 Output: " + '\033[m')
    db = HW2_sql()
    try:
        conn = db.create_connection("Q2")
    except:
        print("Database Creation Error")

    try:
        conn.execute("DROP TABLE IF EXISTS movies;")
        conn.execute("DROP TABLE IF EXISTS movie_cast;")
        conn.execute("DROP TABLE IF EXISTS cast_bio;")
        conn.execute("DROP VIEW IF EXISTS good_collaboration;")
        conn.execute("DROP TABLE IF EXISTS movie_overview;")
    except:
        print("Error in Table Drops")

    try:
        print('\033[32m' + "part ai 1: " + '\033[m' + str(db.part_ai_1(conn)))
        print('\033[32m' + "part ai 2: " + '\033[m' + str(db.part_ai_2(conn)))
    except:
         print("Error in Part a.i")

    try:
        print('\033[32m' + "Row count for Movies Table: " + '\033[m' + str(db.part_aii_1(conn,"data/movies.csv")))
        print('\033[32m' + "Row count for Movie Cast Table: " + '\033[m' + str(db.part_aii_2(conn,"data/movie_cast.csv")))
    except Exception as e:
        raise(e)
        print("Error in part a.ii")

    try:
        print('\033[32m' + "Row count for Cast Bio Table: " + '\033[m' + str(db.part_aiii(conn)))
    except:
        print("Error in part a.iii")

    try:
        print('\033[32m' + "part b 1: " + '\033[m' + db.part_b_1(conn))
        print('\033[32m' + "part b 2: " + '\033[m' + db.part_b_2(conn))
        print('\033[32m' + "part b 3: " + '\033[m' + db.part_b_3(conn))
    except:
        print("Error in part b")

    try:
        print('\033[32m' + "part c: " + '\033[m' + str(db.part_c(conn)))
    except:
        print("Error in part c")

    try:
        print('\033[32m' + "part d: " + '\033[m')
        for line in db.part_d(conn):
            print(line[0],line[1])
    except:
        print("Error in part d")

    try:
        print('\033[32m' + "part e: " + '\033[m')
        for line in db.part_e(conn):
            print(line[0],line[1],line[2])
    except:
        print("Error in part e")

    try:
        print('\033[32m' + "part f: " + '\033[m')
        for line in db.part_f(conn):
            print(line[0],line[1],line[2])
    except:
        print("Error in part f")
    
    try:
        print('\033[32m' + "part g: " + '\033[m' + str(db.part_g(conn)))
        print('\033[32m' + "part g.i: " + '\033[m')
        for line in db.part_gi(conn):
            print(line[0],line[1],line[2])
    except:
        print("Error in part g")

    try:   
        print('\033[32m' + "part h.i: " + '\033[m'+ str(db.part_h(conn,"data/movie_overview.csv")))
        print('\033[32m' + "Count h.ii: " + '\033[m' + str(db.part_hi(conn)))
        print('\033[32m' + "Count h.iii: " + '\033[m' + str(db.part_hii(conn)))
    except:
        print("Error in part h")

    conn.close()
    #################################################################################
    #################################################################################
  
