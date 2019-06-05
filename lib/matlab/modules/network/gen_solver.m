function gen_solver(solver_dir, param)
fields = fieldnames(param.solver);
fid = fopen(solver_dir, 'w');
for idx_field = 1:length(fields)
    if(ischar(param.solver.(fields{idx_field})))
        fprintf(fid, [fields{idx_field} ': ' param.solver.(fields{idx_field}) '\n']);
    elseif(isreal(param.solver.(fields{idx_field})))
        fprintf(fid, [fields{idx_field} ': ' num2str(param.solver.(fields{idx_field})) '\n']);
    end
end
fclose(fid);