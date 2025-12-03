#pragma once

#include <arrow/api.h>
#include <arrow/ipc/api.h>
#include <arrow/io/api.h>
#include <stdexcept>
#include <vector>
#include <string>

#define ARROW_THROW_NOT_OK(expr) \
    do { auto _s = (expr); if (!_s.ok()) throw std::runtime_error(_s.ToString()); } while (0)

/*
 * ArrowIO class for writing data to a file.
 */
class ArrowIO {
public:
    ArrowIO(const std::string& path)
        : m_path(path), m_pool(arrow::default_memory_pool())
    {
        // y : List<List<Float64>>
        auto value_builder_type = arrow::float64();
        auto inner_list_type = arrow::list(value_builder_type);
        auto outer_list_type = arrow::list(inner_list_type);

        m_schema = arrow::schema({
            arrow::field("step", arrow::int64()),
            arrow::field("y", outer_list_type)
        });

        auto file_res = arrow::io::FileOutputStream::Open(m_path);
        if (!file_res.ok())
            throw std::runtime_error("Cannot open trajectory.arrow");

        m_outfile = *file_res;

        auto writer_res = arrow::ipc::MakeStreamWriter(m_outfile, m_schema);
        if (!writer_res.ok())
            throw std::runtime_error("Cannot create IPC StreamWriter");

        m_writer = *writer_res;
    }

    ~ArrowIO() {
        try { close(); } catch(...) {}
    }

    void close() {
        if (m_writer) { ARROW_THROW_NOT_OK(m_writer->Close()); m_writer.reset(); }
        if (m_outfile) { ARROW_THROW_NOT_OK(m_outfile->Close()); m_outfile.reset(); }
    }

    void write_step(std::size_t step_index,
                    std::size_t rows,
                    std::size_t cols,
                    const double* data)
    {
        arrow::Int64Builder step_builder(m_pool);

        arrow::ListBuilder outer_list_builder(
            m_pool,
            std::make_shared<arrow::ListBuilder>(
                m_pool,
                std::make_shared<arrow::DoubleBuilder>(m_pool)
            )
        );

        auto* inner_list_builder =
            static_cast<arrow::ListBuilder*>(outer_list_builder.value_builder());
        auto* value_builder =
            static_cast<arrow::DoubleBuilder*>(inner_list_builder->value_builder());

        // Step index
        ARROW_THROW_NOT_OK(step_builder.Append(static_cast<int64_t>(step_index)));

        // Begin outer list (matrix)
        ARROW_THROW_NOT_OK(outer_list_builder.Append());

        // Column-major indexing: data[c * rows + r]
        for (std::size_t c = 0; c < cols; c++) {
            ARROW_THROW_NOT_OK(inner_list_builder->Append());
            for (std::size_t r = 0; r < rows; r++) {
                ARROW_THROW_NOT_OK(value_builder->Append(data[c * rows + r]));
            }
        }

        // Finalize arrays
        std::shared_ptr<arrow::Array> step_array;
        std::shared_ptr<arrow::Array> y_array;
        ARROW_THROW_NOT_OK(step_builder.Finish(&step_array));
        ARROW_THROW_NOT_OK(outer_list_builder.Finish(&y_array));

        auto batch = arrow::RecordBatch::Make(
            m_schema, 1, {step_array, y_array});

        ARROW_THROW_NOT_OK(m_writer->WriteRecordBatch(*batch));
    }

    std::string path() const { return m_path; }

private:
    std::string m_path;
    arrow::MemoryPool* m_pool;
    std::shared_ptr<arrow::Schema> m_schema;
    std::shared_ptr<arrow::ipc::RecordBatchWriter> m_writer;
    std::shared_ptr<arrow::io::FileOutputStream> m_outfile;
};
