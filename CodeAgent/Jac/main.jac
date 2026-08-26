import from nodes {Plan, Explore, Edit, Verify, Finish, llm}
walker CodeAgent{
    has issue:str;
    can setup with Root entry{
        if [-->] is None{
            pl = Plan();
            ex = Explore();
            ed = Edit();
            ve = Verify();
            fi = Finish();

            here ++> pl;
            pl ++> ex;
            pl ++> fi;
            ex ++> fi;
            ex ++> ed;
            ed ++> ve;
            ve ++> fi;
            ve ++> ed;
        }
    }
    can planning with Plan entry{
        plans:list[str] = here.make_plan(self.issue);
        for plan in plans{
            visit [-->] by llm(
                intent= "Route to the node that can handle this task plan",
            );
        }
    }
}
with entry {
    print("Hello from Jac!");
}
